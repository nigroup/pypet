__author__ = 'Robert Meyer'

import os
import platform
import sys
if (sys.version_info < (2, 7, 0)):
    import unittest2 as unittest
else:
    import unittest
import warnings

import numpy as np
import pandas as pd
from scipy import sparse as spsp
import tables as pt
import logging

from pypet import Trajectory, Parameter, load_trajectory, ArrayParameter, SparseParameter, \
    SparseResult, Result, NNGroupNode, ResultGroup, ConfigGroup, DerivedParameterGroup, \
    ParameterGroup, Environment, pypetconstants, compat, HDF5StorageService
from pypet.tests.testutils.data import TrajectoryComparator
from pypet.tests.testutils.ioutils import make_temp_dir, get_root_logger, \
    parse_args, run_suite, get_log_config, get_log_path
from pypet.utils import ptcompat as ptcompat
from pypet.utils.comparisons import results_equal
import pypet.pypetexceptions as pex
from pypet import new_group as a_new_group


class FakeResult(Result):
    def _store(self):
        raise RuntimeError('I won`t store')


class FakeResult2(Result):
    def __init__(self, full_name, *args, **kwargs):
        super(FakeResult2, self).__init__(full_name, *args, **kwargs)
        self._store_call = 0
    def _store(self):
        res = {}
        if self._store_call == 0:
            res['hey'] = np.ones((10,10))
        if self._store_call > 0:
            res['fail']=FakeResult # This will faile
        self._store_call += 1
        return res


class MyParamGroup(ParameterGroup):
    pass


class StorageTest(TrajectoryComparator):

    tags = 'unittest', 'trajectory', 'hdf5'

    def test_max_depth_loading_and_storing(self):
        filename = make_temp_dir('newassignment.hdf5')
        traj = Trajectory(filename=filename, overwrite_file=True)

        traj.par = Parameter('d1.d2.d3.d4.d5', 55)
        traj.store(max_depth=4)

        traj = load_trajectory(index=-1, filename=filename)
        traj.load(load_data=2)
        self.assertTrue('d3' in traj)
        self.assertFalse('d4' in traj)

        traj = load_trajectory(index=-1, filename=filename, max_depth=3)
        self.assertTrue('d2' in traj)
        self.assertFalse('d3' in traj)

        traj.par.remove(recursive=True)
        traj.dpar = Parameter('d1.d2.d3.d4.d5', 123)

        traj.dpar.store_child('d1', recursive=True, max_depth=3)
        traj.dpar.remove_child('d1', recursive=True)

        self.assertTrue('d1' not in traj)
        traj.dpar.load_child('d1', recursive=True)

        self.assertTrue('d3' in traj)
        self.assertTrue('d4' not in traj)

        traj.dpar.remove_child('d1', recursive=True)
        self.assertTrue('d1' not in traj)
        traj.dpar.load_child('d1', recursive=True, max_depth=2)

        self.assertTrue('d2' in traj)
        self.assertTrue('d3' not in traj)

        traj.dpar = Parameter('l1.l2.l3.l4.l5', 123)
        traj.dpar.store(recursive=True, max_depth=0)
        self.assertFalse(traj.dpar.l1._stored)

        traj.dpar.store(recursive=True, max_depth=4)
        traj.dpar.remove()
        self.assertTrue('l1' not in traj)
        traj.dpar.load(recursive=True)
        self.assertTrue('l4' in traj)
        self.assertTrue('l5' not in traj)

        traj.dpar.remove()
        self.assertTrue('l1' not in traj)
        traj.dpar.load(recursive=True, max_depth=3)
        self.assertTrue('l3' in traj)
        self.assertTrue('l4' not in traj)

    def test_file_size_many_params(self):
        filename = make_temp_dir('filesize.hdf5')
        traj = Trajectory(filename=filename, overwrite_file=True, add_time=False)
        npars = 700
        traj.store()
        for irun in range(npars):
            par = traj.add_parameter('test.test%d' % irun, 42+irun, comment='duh!')
            traj.store_item(par)


        size =  os.path.getsize(filename)
        size_in_mb = size/1000000.
        get_root_logger().info('Size is %sMB' % str(size_in_mb))
        self.assertTrue(size_in_mb < 10.0, 'Size is %sMB > 10MB' % str(size_in_mb))

    def test_loading_explored_parameters(self):

        filename = make_temp_dir('load_explored.hdf5')
        traj = Trajectory(filename=filename, overwrite_file=True, add_time=False)
        traj.par.x = Parameter('x', 42, comment='answer')
        traj.explore({'x':[1,2,3,4]})
        traj.store()
        name = traj.name

        traj = Trajectory(filename=filename, add_time=False)
        traj.load()
        x = traj.get('x')
        self.assertIs(x, traj._explored_parameters['parameters.x'])

    def test_loading_and_storing_empty_containers(self):
        filename = make_temp_dir('empty_containers.hdf5')
        traj = Trajectory(filename=filename, add_time=True)

        # traj.f_add_parameter('empty.dict', {})
        # traj.f_add_parameter('empty.list', [])
        traj.add_parameter(ArrayParameter, 'empty.tuple', ())
        traj.add_parameter(ArrayParameter, 'empty.array', np.array([], dtype=float))

        spsparse_csc = spsp.csc_matrix((2,10))
        spsparse_csr = spsp.csr_matrix((6660,660))
        spsparse_bsr = spsp.bsr_matrix((3330,2220))
        spsparse_dia = spsp.dia_matrix((1230,1230))

        traj.add_parameter(SparseParameter, 'empty.csc', spsparse_csc)
        traj.add_parameter(SparseParameter, 'empty.csr', spsparse_csr)
        traj.add_parameter(SparseParameter, 'empty.bsr', spsparse_bsr)
        traj.add_parameter(SparseParameter, 'empty.dia', spsparse_dia)

        traj.add_result(SparseResult, 'empty.all', dict={}, list=[],
                          series = pd.Series(),
                          frame = pd.DataFrame(),
                          panel = pd.Panel(),
                          **traj.par.to_dict(short_names=True, fast_access=True))

        traj.store()

        newtraj = load_trajectory(index=-1, filename=filename)

        newtraj.load(load_data=2)

        epg = newtraj.par.empty
        self.assertTrue(type(epg.tuple) is tuple)
        self.assertTrue(len(epg.tuple) == 0)

        self.assertTrue(type(epg.array) is np.ndarray)
        self.assertTrue(epg.array.size == 0)

        self.assertTrue(spsp.isspmatrix_csr(epg.csr))
        self.assertTrue(epg.csr.size == 0)

        self.assertTrue(spsp.isspmatrix_csc(epg.csc))
        self.assertTrue(epg.csc.size == 0)

        self.assertTrue(spsp.isspmatrix_bsr(epg.bsr))
        self.assertTrue(epg.bsr.size == 0)

        self.assertTrue(spsp.isspmatrix_dia(epg.dia))
        self.assertTrue(epg.dia.size == 0)

        self.compare_trajectories(traj, newtraj)


    def test_new_assignment_method(self):
        filename = make_temp_dir('newassignment.hdf5')
        traj = Trajectory(filename=filename, add_time=True)

        traj.lazy_adding = True
        comment = 'A number'
        traj.par.x = 44, comment

        self.assertTrue(traj.get('x').comment == comment)

        traj.par.iamgroup = a_new_group

        self.assertTrue(isinstance(traj.iamgroup, ParameterGroup))

        traj.lazy_adding = False
        traj.x = 45
        self.assertTrue(traj.par.get('x').get() == 45)

        self.assertTrue(isinstance(traj.get('x'), Parameter))

        traj.f = Parameter('lll', 444, 'lll')

        self.assertTrue(traj.get('f').name == 'f')

        traj.lazy_adding = True
        traj.res.k = 22, 'Hi'
        self.assertTrue(isinstance(traj.get('k'), Result))
        self.assertTrue(traj.get('k')[1] == 'Hi')

        with self.assertRaises(AttributeError):
            traj.res.k = 33, 'adsd'

        conf = traj.conf
        with self.assertRaises(AttributeError):
            conf = traj.conf.jjjj
        traj.set_properties(fast_access=True)


        traj.crun = 43, 'JJJ'
        self.assertTrue(traj.run_A[0] == 43)

        with self.assertRaises(AttributeError):
            traj.set_properties(j=7)

        with self.assertRaises(AttributeError):
            traj.set_properties(depth=7)

        traj.hui = (('444', 'kkkk',), 'l')



        self.assertTrue(traj.get('hui')[1] == 'l')

        with self.assertRaises(AttributeError):
            traj.hui = ('445', 'kkkk',)

        traj.get('hui').set(('445', 'kkkk',))

        self.assertTrue(traj.get('hui')[1] == 'l')

        self.assertTrue(traj.hui[0] == ('445', 'kkkk',))

        traj.add_link('klkikju', traj.par) # for shizzle


        traj.meee = Result('h', 43, hui = 3213, comment='du')

        self.assertTrue(traj.meee.h.h == 43)

        with self.assertRaises(TypeError):
            traj.par.mu = NNGroupNode('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.res.mu = NNGroupNode('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.conf.mu = NNGroupNode('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.dpar.mu = NNGroupNode('jj', comment='mi')

        with self.assertRaises(TypeError):
            traj.par.mu = ResultGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.dpar.mu = ResultGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.conf.mu = ResultGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.mu = ResultGroup('jj', comment='mi')

        with self.assertRaises(TypeError):
            traj.par.mu = ConfigGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.dpar.mu = ConfigGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.res.mu = ConfigGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.mu = ConfigGroup('jj', comment='mi')

        with self.assertRaises(TypeError):
            traj.par.mu = DerivedParameterGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.conf.mu = DerivedParameterGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.res.mu = DerivedParameterGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.mu = DerivedParameterGroup('jj', comment='mi')

        with self.assertRaises(TypeError):
            traj.dpar.mu = ParameterGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.res.mu = ParameterGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.conf.mu = ParameterGroup('jj', comment='mi')
        with self.assertRaises(TypeError):
            traj.mu = ParameterGroup('jj', comment='mi')

        traj.par.mu = ParameterGroup('jj', comment='mi')
        traj.res.mus = ResultGroup('jj', comment='mi')
        traj.mu = NNGroupNode('jj')
        cg = ConfigGroup('a.g')
        traj.conf.a = cg

        self.assertTrue(traj.get('conf.a.a.g', shortcuts=False) is cg)

        dg = DerivedParameterGroup('ttt')
        traj.dpar.ttt = dg

        self.assertTrue(traj.get('dpar.ttt', shortcuts=False) is dg)

        traj.mylink = traj.par

        self.assertTrue(traj.mylink is traj.par)

        traj.vvv = NNGroupNode('', comment='kkk')

        self.assertTrue(traj.vvv.full_name == 'vvv')

        self.assertTrue(traj.par.mu.name == 'mu')

        traj.rrr = MyParamGroup('ff')

        traj.par.g = MyParamGroup('')

        pg = traj.add_parameter_group(comment='gg', full_name='me')
        self.assertTrue(traj.par.me is pg)

        traj.store()

        traj = load_trajectory(index=-1, filename=filename, dynamic_imports=MyParamGroup)

        self.assertTrue(isinstance(traj.rrr, NNGroupNode))
        self.assertTrue(isinstance(traj.rrr.ff, MyParamGroup))
        self.assertTrue(isinstance(traj.par.g, MyParamGroup))

        traj.par = Parameter('hiho', 42, comment='you')
        traj.par = Parameter('g1.g2.g3.g4.g5', 43)

        self.assertTrue(traj.hiho == 42)
        self.assertTrue(isinstance(traj.par.g1, ParameterGroup ))
        self.assertTrue(isinstance(traj.par.g3, ParameterGroup ))
        self.assertTrue(traj.g3.g5 == 43)


    def test_shortenings_of_names(self):
        traj = Trajectory(filename=make_temp_dir('testshortening.hdf5'), add_time=True)
        traj.aconf('g', 444)
        self.assertTrue(isinstance(traj.get('g'), Parameter))
        self.assertTrue(traj.conf.g == 444)

        traj.apar('g', 444)
        self.assertTrue(isinstance(traj.par.get('g'), Parameter))
        self.assertTrue(traj.par.g == 444)

        traj.adpar('g', 445)
        self.assertTrue(isinstance(traj.derived_parameters.get('g'), Parameter))
        self.assertTrue(traj.dpar.g == 445)

        traj.ares('g', 454)
        self.assertTrue(isinstance(traj.res.get('g'), Result))
        self.assertTrue(traj.res.g == 454)


    def test_storage_service_errors(self):

        traj = Trajectory(filename=make_temp_dir('testnoservice.hdf5'), add_time=True)

        traj_name = traj.name

        # you cannot store stuff before the trajectory was stored once:
        with self.assertRaises(ValueError):
            traj.storage_service.store('FAKESERVICE', self, trajectory_name = traj.name)

        traj.store()

        with self.assertRaises(ValueError):
            traj.storage_service.store('FAKESERVICE', self, trajectory_name = 'test')

        with self.assertRaises(pex.NoSuchServiceError):
            traj.storage_service.store('FAKESERVICE', self, trajectory_name = traj.name)

        with self.assertRaises(ValueError):
            traj.load(name='test', index=1)

        with self.assertRaises(RuntimeError):
            traj.storage_service.store('LIST', [('LEAF',None,None,None,None)],
                                         trajectory_name = traj.name)

        with self.assertRaises(ValueError):
            traj.load(index=9999)

        with self.assertRaises(ValueError):
            traj.load(name='Non-Existising-Traj')

    def test_storing_and_loading_groups(self):
        filename = make_temp_dir('grpgrp.hdf5')
        traj = Trajectory(name='traj', add_time=True, filename=filename)
        res=traj.add_result('aaa.bbb.ccc.iii', 42, 43, comment=7777 * '6')
        traj.ccc.annotations['gg']=4
        res=traj.add_result('aaa.ddd.eee.jjj', 42, 43, comment=777 * '6')
        traj.ccc.annotations['j'] = 'osajdsojds'
        traj.store(only_init=True)
        traj.store_item('aaa', recursive=True)
        newtraj = load_trajectory(traj.name, filename=filename, load_all=2)

        self.compare_trajectories(traj, newtraj)

        traj.iii.set(55)

        self.assertFalse(results_equal(traj.iii, newtraj.iii))

        traj.aaa.store(recursive=True, store_data=3)

        newtraj.bbb.load(recursive=True, load_data=3)

        self.compare_trajectories(traj, newtraj)

        traj.ccc.annotations['gg'] = 5
        traj.load(load_data=3)
        self.assertTrue(traj.ccc.annotations['gg'] == 4)
        traj.ccc.annotations['gg'] = 5
        traj.store(store_data=3)
        newtraj.load(load_data=2)
        self.assertTrue(newtraj.ccc.annotations['gg'] == 4)
        newtraj.load(load_data=3)
        self.assertTrue(newtraj.ccc.annotations['gg'] == 5)

        traj.ccc.add_link('link', res)
        traj.store_item(traj.ccc, store_data=3, with_links=False)

        newtraj.load(load_data=3)
        self.assertTrue('link' not in newtraj.ccc)

        traj.store_item(traj.ccc, store_data=3, with_links=True, recursive=True)

        newtraj.load_item(newtraj.ccc, with_links=False, recursive=True)
        self.assertTrue('link' not in newtraj.ccc)

        newtraj.load_item(newtraj.ccc, recursive=True)
        self.assertTrue('link' in newtraj.ccc)


    def test_store_overly_long_comment(self):
        filename = make_temp_dir('remove_errored.hdf5')
        traj = Trajectory(name='traj', add_time=True, filename=filename)
        res=traj.add_result('iii', 42, 43, comment=7777 * '6')
        traj.store()
        traj.remove_child('results', recursive=True)
        traj.load_child('results', recursive=True)
        self.assertTrue(traj.iii.comment == 7777 * '6')

    def test_removal_of_error_parameter(self):

        filename = make_temp_dir('remove_errored.hdf5')
        traj = Trajectory(name='traj', add_time=True, filename=filename)
        traj.add_result('iii', 42)
        traj.add_result(FakeResult, 'j.j.josie', 43)

        file = traj.storage_service.filename
        traj.store(only_init=True)
        with self.assertRaises(RuntimeError):
            traj.store()

        with ptcompat.open_file(file, mode='r') as fh:
            jj = ptcompat.get_node(fh, where='/%s/results/j/j' % traj.name)
            self.assertTrue('josie' not in jj)

        traj.j.j.remove_child('josie')
        traj.j.j.add_result(FakeResult2, 'josie2', 444)

        traj.store()
        with self.assertRaises(pex.NoSuchServiceError):
            traj.store_child('results', recursive=True)

        with ptcompat.open_file(file, mode='r') as fh:
            jj = ptcompat.get_node(fh, where='/%s/results/j/j' % traj.name)
            self.assertTrue('josie2' in jj)
            josie2 = ptcompat.get_child(jj, 'josie2')
            self.assertTrue('hey' in josie2)
            self.assertTrue('fail' not in josie2)


    def test_maximum_overview_size(self):

        filename = make_temp_dir('maxisze.hdf5')

        env = Environment(trajectory='Testmigrate', filename=filename,

                          log_config=get_log_config(), add_time=True)

        traj = env.trajectory
        for irun in range(pypetconstants.HDF5_MAX_OVERVIEW_TABLE_LENGTH):
            traj.add_parameter('f%d.x' % irun, 5)

        traj.store()


        store = ptcompat.open_file(filename, mode='r+')
        table = ptcompat.get_child(store.root,traj.name).overview.parameters_overview
        self.assertEquals(table.nrows, pypetconstants.HDF5_MAX_OVERVIEW_TABLE_LENGTH)
        store.close()

        for irun in range(pypetconstants.HDF5_MAX_OVERVIEW_TABLE_LENGTH,
                  2*pypetconstants.HDF5_MAX_OVERVIEW_TABLE_LENGTH):
            traj.add_parameter('f%d.x' % irun, 5)

        traj.store()

        store = ptcompat.open_file(filename, mode='r+')
        table = ptcompat.get_child(store.root,traj.name).overview.parameters_overview
        self.assertEquals(table.nrows, pypetconstants.HDF5_MAX_OVERVIEW_TABLE_LENGTH)
        store.close()

        env.disable_logging()

    def test_overwrite_annotations_and_results(self):

        filename = make_temp_dir('overwrite.hdf5')

        env = Environment(trajectory='testoverwrite', filename=filename,
                          log_config=get_log_config(), overwrite_file=True)

        traj = env.traj

        traj.add_parameter('grp.x', 5, comment='hi')
        traj.grp.comment='hi'
        traj.grp.annotations['a'] = 'b'

        traj.store()

        traj.remove_child('parameters', recursive=True)

        traj.load(load_data=2)

        self.assertTrue(traj.x == 5)
        self.assertTrue(traj.grp.comment == 'hi')
        self.assertTrue(traj.grp.annotations['a'] == 'b')

        traj.get('x').unlock()
        traj.grp.x = 22
        traj.get('x').comment='hu'
        traj.grp.annotations['a'] = 'c'
        traj.grp.comment = 'hu'

        traj.store_item(traj.get('x'), store_data=3)
        traj.store_item(traj.grp, store_data=3)

        traj.remove_child('parameters', recursive=True)

        traj.load(load_data=2)

        self.assertTrue(traj.x == 22)
        self.assertTrue(traj.grp.comment == 'hu')
        self.assertTrue(traj.grp.annotations['a'] == 'c')

        env.disable_logging()


    def test_migrations(self):

        traj = Trajectory(name='Testmigrate', filename=make_temp_dir('migrate.hdf5'),
                          add_time=True)

        traj.add_result('I.am.a.mean.resu', 42, comment='Test')
        traj.add_derived_parameter('ffa', 42)

        traj.store()

        new_file = make_temp_dir('migrate2.hdf5')
        traj.migrate(filename=new_file)

        traj.store()

        new_traj = Trajectory()

        new_traj.migrate(new_name=traj.name, filename=new_file, in_store=True)

        new_traj.auto_load=True

        self.assertTrue(new_traj.results.I.am.a.mean.resu == 42)

    def test_wildcard_search(self):

        traj = Trajectory(name='Testwildcard', filename=make_temp_dir('wilcard.hdf5'),
                          add_time=True)

        traj.add_parameter('expl', 2)
        traj.explore({'expl':[1,2,3,4]})

        traj.add_result('wc2test.$.hhh', 333)
        traj.add_leaf('results.wctest.run_00000000.jjj', 42)
        traj.add_result('results.wctest.run_00000001.jjj', 43)
        traj.add_result('results.wctest.%s.jjj' % traj.wildcard('$', -1), 43)

        traj.as_run = 1

        self.assertTrue(traj.results.wctest['$'].jjj==43)
        self.assertTrue(traj.results.wc2test.crun.hhh==333)

        traj.store()

        get_root_logger().info('Removing child1')

        traj.remove_child('results', recursive=True)

        get_root_logger().info('Doing auto-load')
        traj.auto_load = True

        self.assertTrue(traj.results.wctest['$'].jjj==43)
        self.assertTrue(traj.results.wc2test.crun.hhh==333)

        get_root_logger().info('Removing child2')

        traj.remove_child('results', recursive=True)

        get_root_logger().info('auto-loading')
        traj.auto_load = True

        self.assertTrue(traj.results.wctest[-1].jjj==43)
        self.assertTrue(traj.results.wc2test[-1].hhh==333)

        get_root_logger().info('Removing child3')
        traj.remove_child('results', recursive=True)

        get_root_logger().info('auto-loading')
        traj.auto_load = True

        self.assertTrue(traj.results.wctest[1].jjj==43)
        self.assertTrue(traj.results.wc2test[-1].hhh==333)

        get_root_logger().info('Done with wildcard test')

    def test_store_and_load_large_dictionary(self):
        traj = Trajectory(name='Testlargedict', filename=make_temp_dir('large_dict.hdf5'),
                          add_time=True)

        large_dict = {}

        for irun in range(1025):
            large_dict['item_%d' % irun] = irun

        large_dict2 = {}

        for irun in range(33):
            large_dict2['item_%d' % irun] = irun

        traj.add_result('large_dict', large_dict, comment='Huge_dict!')
        traj.add_result('large_dict2', large_dict2, comment='Not so large dict!')

        traj.store()

        traj_name = traj.name

        traj2 = Trajectory(filename=make_temp_dir('large_dict.hdf5'),
                           add_time=True)

        traj2.load(name=traj_name, load_data=2)

        self.compare_trajectories(traj, traj2)


    def test_auto_load(self):


        traj = Trajectory(name='Testautoload', filename=make_temp_dir('autoload.hdf5'),
                          add_time=True)

        traj.auto_load = True

        traj.add_result('I.am.$.a.mean.resu', 42, comment='Test')

        traj.add_derived_parameter('ffa', 42)

        traj.store()

        ffa=traj.get('ffa')
        ffa.unlock()
        ffa.empty()

        self.assertTrue(ffa.is_empty())

        traj.remove_child('results', recursive=True)

        # check auto load
        val = traj.res.I.am.crun.a.mean.resu

        self.assertTrue(val==42)

        val = traj.ffa

        self.assertTrue(val==42)

        with self.assertRaises(pex.DataNotInStorageError):
            traj.kdsfdsf

    def test_get_default(self):


        traj = Trajectory(name='Testgetdefault', filename=make_temp_dir('autoload.hdf5'),
                          add_time=True)

        traj.auto_load = True

        traj.add_result('I.am.$.a.mean.resu', 42, comment='Test')

        val = traj.get_default('jjjjjjjjjj', 555)
        self.assertTrue(val==555)

        traj.store()

        traj.remove_child('results', recursive=True)




        val = traj.get_default('res.I.am.crun.a.mean.answ', 444, auto_load=True)

        self.assertTrue(val==444)

        val = traj.get_default('res.I.am.crun.a.mean.resu', auto_load=True, fast_access=True)

        self.assertTrue(val==42)

        with self.assertRaises(Exception):
            traj.kdsfdsf


    def test_version_mismatch(self):
        traj = Trajectory(name='TestVERSION', filename=make_temp_dir('testversionmismatch.hdf5'),
                          add_time=True)

        traj.add_parameter('group1.test',42)

        traj.add_result('testres', 42)

        traj.group1.set_annotations(Test=44)

        traj._version='0.1a.1'

        traj.store()

        traj2 = Trajectory(name=traj.name, add_time=False,
                           filename=make_temp_dir('testversionmismatch.hdf5'))

        with self.assertRaises(pex.VersionMismatchError):
            traj2.load(load_parameters=2, load_results=2)

        traj2.load(load_parameters=2, load_results=2, force=True)

        self.compare_trajectories(traj,traj2)

        get_root_logger().info('Mismatch testing done!')

    def test_fail_on_wrong_kwarg(self):
        with self.assertRaises(ValueError):
            filename = 'testsfail.hdf5'
            env = Environment(filename=make_temp_dir(filename),
                          log_stdout=True,
                          log_config=get_log_config(),
                          logger_names=('STDERROR', 'STDOUT'),
                          foo='bar')

    def test_no_run_information_loading(self):
        filename = make_temp_dir('testnoruninfo.hdf5')
        traj = Trajectory(name='TestDelete',
                          filename=filename,
                          add_time=True)

        length = 100000
        traj.lazy_adding = True
        traj.par.x = 42
        traj.explore({'x': range(length)})

        traj.store()

        traj = load_trajectory(index=-1, filename=filename, with_run_information=False)
        self.assertEqual(len(traj), length)
        self.assertEqual(len(traj._run_information), 1)

    def test_delete_whole_subtrees(self):
        filename = make_temp_dir('testdeltree.hdf5')
        traj = Trajectory(name='TestDelete',
                          filename=filename, large_overview_tables=True,
                          add_time=True)

        res = traj.add_result('mytest.yourtest.test', a='b', c='d')
        dpar = traj.add_derived_parameter('mmm.gr.dpdp', 666)


        res = traj.add_result('hhh.ll', a='b', c='d')
        res = traj.add_derived_parameter('hhh.gg', 555)

        traj.store()

        with ptcompat.open_file(filename) as fh:
            daroot = ptcompat.get_child(fh.root, traj.name)
            dpar_table = daroot.overview.derived_parameters_overview
            self.assertTrue(len(dpar_table) == 2)
            res_table = daroot.overview.results_overview
            self.assertTrue((len(res_table)) == 2)

        with self.assertRaises(TypeError):
            traj.remove_item(traj.yourtest)

        with self.assertRaises(TypeError):
            traj.delete_item(traj.yourtest)

        traj.remove_item(traj.yourtest, recursive=True)

        self.assertTrue('mytest' in traj)
        self.assertTrue('yourtest' not in traj)

        traj.load(load_data=2)

        self.assertTrue('yourtest.test' in traj)

        traj.delete_item(traj.yourtest, recursive=True, remove_from_trajectory=True)
        traj.delete_item(traj.mmm, recursive=True, remove_from_trajectory=True)

        traj.load(load_data=2)

        self.assertTrue('yourtest.test' not in traj)
        self.assertTrue('yourtest' not in traj)

        with ptcompat.open_file(filename) as fh:
            daroot = ptcompat.get_child(fh.root, traj.name)
            dpar_table = daroot.overview.derived_parameters_overview
            self.assertTrue(len(dpar_table) == 2)
            res_table = daroot.overview.results_overview
            self.assertTrue((len(res_table)) == 2)

        traj.add_parameter('ggg', 43)
        traj.add_parameter('hhh.mmm', 45)
        traj.add_parameter('jjj', 55)
        traj.add_parameter('hhh.nnn', 55555)

        traj.explore({'ggg':[1,2,3]})

        traj.store()

        with ptcompat.open_file(filename) as fh:
            daroot = ptcompat.get_child(fh.root, traj.name)
            par_table = daroot.overview.parameters_overview
            self.assertTrue(len(par_table) == 4)

        traj.delete_item('par.hhh', recursive=True, remove_from_trajectory=True)

        traj.add_parameter('saddsdfdsfd', 111)
        traj.store()

        with ptcompat.open_file(filename) as fh:
            daroot = ptcompat.get_child(fh.root, traj.name)
            par_table = daroot.overview.parameters_overview
            self.assertTrue(len(par_table) == 5)

        # with self.assertRaises(TypeError):
        #     # We cannot delete something containing an explored parameter
        #     traj.f_delete_item('par', recursive=True)

        # with self.assertRaises(TypeError):
        #     traj.f_delete_item('ggg')

    def test_delete_links(self):
        traj = Trajectory(name='TestDelete',
                          filename=make_temp_dir('testpartiallydel.hdf5'),
                          add_time=True)

        res = traj.add_result('mytest.test', a='b', c='d')

        traj.add_link('x.y', res)
        traj.add_link('x.g.h', res)

        traj.store()

        traj.remove_child('x', recursive=True)
        traj.load()

        self.assertEqual(traj.x.y.a, traj.test.a)
        self.assertEqual(traj.x.g.h.a, traj.test.a)

        traj.delete_link('x.y', remove_from_trajectory=True)
        traj.delete_link((traj.x.g, 'h'), remove_from_trajectory=True)

        traj.load()

        with self.assertRaises(AttributeError):
            traj.x.g.h


    def test_partially_delete_stuff(self):
        traj = Trajectory(name='TestDelete',
                          filename=make_temp_dir('testpartiallydel.hdf5'),
                          add_time=True)

        res = traj.add_result('mytest.test', a='b', c='d')

        traj.store()

        self.assertTrue('a' in res)
        traj.delete_item(res, delete_only=['a'], remove_from_item=True)

        self.assertTrue('c' in res)
        self.assertTrue('a' not in res)

        res['a'] = 'offf'

        self.assertTrue('a' in res)

        traj.load(load_results=3)

        self.assertTrue('a' not in res)
        self.assertTrue('c' in res)

        traj.delete_item(res, remove_from_trajectory=True)

        self.assertTrue('results' in traj)
        self.assertTrue(res not in traj)

    def test_throw_warning_if_old_kw_is_used(self):
        pass

        filename = make_temp_dir('hdfwarning.hdf5')

        with warnings.catch_warnings(record=True) as w:

            env = Environment(trajectory='test', filename=filename,
                              dynamically_imported_classes=[],
                              log_config=get_log_config(), add_time=True)

        with warnings.catch_warnings(record=True) as w:
            traj = Trajectory(dynamically_imported_classes=[])

        traj = env.trajectory
        traj.store()

        with warnings.catch_warnings(record=True) as w:
            traj.load(dynamically_imported_classes=[])

        env.disable_logging()

    def test_overwrite_stuff(self):
        traj = Trajectory(name='TestOverwrite', filename=make_temp_dir('testowrite.hdf5'),
                          add_time=True)

        res = traj.add_result('mytest.test', a='b', c='d')

        traj.store()

        res['a'] = np.array([1,2,3])
        res['c'] = 123445

        traj.store_item(res, overwrite='a', complevel=4)

        # Should emit a warning
        traj.store_item(res, overwrite=['a', 'b'])

        traj.load(load_results=3)

        res = traj.test

        self.assertTrue((res['a']==np.array([1,2,3])).all())
        self.assertTrue(res['c']=='d')

        res['c'] = 123445

        traj.store_item(res, store_data=3)
        res.empty()

        traj.load(load_results=3)

        self.assertTrue(traj.test['c']==123445)

    def test_loading_as_new(self):
        filename = make_temp_dir('asnew.h5')
        traj = Trajectory(name='TestPartial', filename=filename, add_time=True)

        traj.add_parameter('x', 3)
        traj.add_parameter('y', 2)

        traj.explore({'x': [12,3,44], 'y':[1,23,4]})

        traj.store()

        traj = load_trajectory(name=traj.name, filename=filename)

        with self.assertRaises(TypeError):
            traj.shrink()

        traj = load_trajectory(name=traj.name, filename=filename, as_new=True,
                               new_name='TestTraj', add_time=False)

        self.assertTrue(traj.name == 'TestTraj')

        self.assertTrue(len(traj) == 3)

        traj.shrink()

        self.assertTrue(len(traj) == 1)


    def test_partial_loading(self):
        traj = Trajectory(name='TestPartial', filename=make_temp_dir('testpartially.hdf5'),
                          add_time=True)

        res = traj.add_result('mytest.test', a='b', c='d')

        traj.store()

        traj.remove_child('results', recursive=True)

        traj.load_skeleton()

        traj.load_item(traj.test, load_only=['a', 'x'])

        self.assertTrue('a' in traj.test)
        self.assertTrue('c' not in traj.test)

        traj.remove_child('results', recursive=True)

        traj.load_skeleton()

        load_except= ['c', 'd']
        traj.load_item(traj.test, load_except=load_except)

        self.assertTrue(len(load_except)==2)

        self.assertTrue('a' in traj.test)
        self.assertTrue('c' not in traj.test)

        with self.assertRaises(ValueError):
            traj.load_item(traj.test, load_except=['x'], load_only=['y'])


    def test_hdf5_settings_and_context(self):

        filename = make_temp_dir('hdfsettings.hdf5')
        with Environment('testraj', filename=filename,
                         add_time=True,
                         comment='',
                         dynamic_imports=None,
                         log_config=None,
                         multiproc=False,
                         ncores=3,
                         wrap_mode=pypetconstants.WRAP_MODE_LOCK,
                         resumable=False,
                         use_hdf5=True,
                         complevel=4,
                         complib='zlib',
                         shuffle=True,
                         fletcher32=True,
                         pandas_format='t',
                         pandas_append=True,
                         purge_duplicate_comments=True,
                         summary_tables=True,
                         small_overview_tables=True,
                         large_overview_tables=True,
                         results_per_run=19,
                         derived_parameters_per_run=17) as env:

            traj = env.trajectory

            traj.store()

            hdf5file = pt.openFile(filename=filename)

            table= hdf5file.root._f_getChild(traj.name)._f_getChild('overview')._f_getChild('hdf5_settings')

            row = table[0]

            self.assertTrue(row['complevel'] == 4)

            self.assertTrue(row['complib'] == compat.tobytes('zlib'))

            self.assertTrue(row['shuffle'])
            self.assertTrue(row['fletcher32'])
            self.assertTrue(row['pandas_format'] == compat.tobytes('t'))

            for attr_name, table_name in HDF5StorageService.NAME_TABLE_MAPPING.items():
                self.assertTrue(row[table_name])

            self.assertTrue(row['purge_duplicate_comments'])
            self.assertTrue(row['results_per_run']==19)
            self.assertTrue(row['derived_parameters_per_run'] == 17)

            hdf5file.close()


    def test_store_items_and_groups(self):

        traj = Trajectory(name='testtraj', filename=make_temp_dir('teststoreitems.hdf5'),
                          add_time=True)

        traj.store()

        traj.add_parameter('group1.test',42, comment= 'TooLong' * pypetconstants.HDF5_STRCOL_MAX_COMMENT_LENGTH)

        traj.add_result('testres', 42)

        traj.group1.set_annotations(Test=44)

        traj.store_items(['test','testres','group1'])


        traj2 = Trajectory(name=traj.name, add_time=False,
                           filename=make_temp_dir('teststoreitems.hdf5'))

        traj2.load(load_parameters=2, load_results=2)

        traj.add_result('Im.stored.along.a.path', 43)
        traj.Im['stored'].along.annotations['wtf'] =4444
        traj.res.store_child('Im.stored.along.a.path')


        traj2.res.load_child('Im.stored.along.a.path', load_data=2)

        self.compare_trajectories(traj,traj2)


if __name__ == '__main__':
    opt_args = parse_args()
    run_suite(**opt_args)