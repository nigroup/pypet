__author__ = 'Robert Meyer'

from pypet import Trajectory, pypetexceptions, PickleResult
from pypet.tests.testutils.data import TrajectoryComparator
from pypet.tests.testutils.ioutils import make_temp_dir, run_suite, parse_args
import pypet.pypetexceptions as pex


class LinkTrajectoryTests(TrajectoryComparator):

    tags = 'unittest', 'links'

    def test_iteration_failure(self):
        traj = Trajectory()

        traj.add_parameter_group('test.test3')
        traj.add_parameter_group('test2')
        traj.test2.add_link(traj.test3)

        with self.assertRaises(pex.NotUniqueNodeError):
            traj.test3

    def test_link_creation(self):

        traj = Trajectory()

        traj.add_parameter_group('test.test3')
        traj.add_parameter_group('test2')

        with self.assertRaises(AttributeError):
            traj.par.add_link('test', traj.test2)

        with self.assertRaises(ValueError):
            traj.add_link('parameters', traj.test)

        with self.assertRaises(ValueError):
            traj.add_link('kkkk', PickleResult('fff', 555))

        traj.test.add_link('circle1' , traj.test2)
        traj.test2.add_link('circle2' , traj.test)

        self.assertTrue(traj.test.circle1.circle2.circle1.circle2 is traj.test)


        traj.add_link('hh', traj.test)

        traj.par.add_link('overview', traj.test)
        with self.assertRaises(ValueError):
            traj.add_link('overview', traj.test)

        with self.assertRaises(ValueError):
            traj.par.add_link('gg', traj)

        with self.assertRaises(AttributeError):
            traj.add_parameter('test.circle1.testy', 33)

        traj.par.add_link('gg', traj.circle1)
        self.assertTrue(traj.gg is traj.test2)
        self.assertTrue(traj.test2.test3 is traj.par.test.test3)

        traj.add_link(traj.test3)
        self.assertTrue('test3' in traj._links)

    def test_not_getting_links(self):
        traj = Trajectory()

        traj.add_parameter_group('test.test3')
        traj.add_parameter_group('test2')

        traj.test.add_link('circle1' , traj.test2)
        traj.test2.add_link('circle2' , traj.test)

        self.assertTrue(traj.test.circle1 is traj.test2)

        traj.with_links = False

        with self.assertRaises(AttributeError):
            traj.test.circle1

        found = False
        for item in traj.test.iter_nodes(recursive=True, with_links=True):
            if item is traj.test2:
                found=True

        self.assertTrue(found)

        for item in traj.test.iter_nodes(recursive=True, with_links=False):
            if item is traj.test2:
                self.assertTrue(False)

        traj.with_links=True
        self.assertTrue('circle1' in traj)
        self.assertFalse(traj.contains('circle1', with_links=False))

        self.assertTrue('circle1' in traj.test)
        self.assertFalse(traj.test.contains('circle1', with_links=False))

        self.assertTrue(traj.test2.test3 is traj.par.test.test3)
        traj.with_links = False
        with self.assertRaises(AttributeError):
            traj.test2.test3

        traj.with_links = True
        self.assertTrue(traj['test2.test3'] is traj.test3)

        with self.assertRaises(AttributeError):
            traj.get('test2.test3', with_links=False)

    def test_link_of_link(self):

        traj = Trajectory()

        traj.add_parameter_group('test')
        traj.add_parameter_group('test2')

        traj.test.add_link('circle1' , traj.test2)
        traj.test2.add_link('circle2' , traj.test)
        traj.test.add_link('circle2' , traj.test.circle1.circle2)


        self.assertTrue(traj.test.circle2 is traj.test)

    def test_link_removal(self):
        traj = Trajectory()

        traj.add_parameter_group('test')
        traj.add_parameter_group('test2')

        traj.test.add_link('circle1' , traj.test2)
        traj.test2.add_link('circle2' , traj.test)

        self.assertTrue('circle1' in traj)
        traj.circle1.circle2.remove_link('circle1')
        self.assertTrue('circle1' not in traj.circle2)

        with self.assertRaises(AttributeError):
            traj.test.circle1

        with self.assertRaises(ValueError):
            traj.test.remove_link('circle1')

        traj.test2.remove_child('circle2')

        self.assertTrue('circle2' not in traj)

    def test_storage_and_loading(self):
        filename = make_temp_dir('linktest.hdf5')
        traj = Trajectory(filename=filename)

        traj.add_parameter_group('test')
        traj.add_parameter_group('test2')
        res= traj.add_result('kk', 42)

        traj.par.add_link('gg', res)

        traj.add_link('hh', res)
        traj.add_link('jj', traj.par)
        traj.add_link('ii', res)

        traj.test.add_link('circle1' , traj.test2)
        traj.test2.add_link('circle2' , traj.test)
        traj.test.add_link('circle2' , traj.test.circle1.circle2)

        traj.add_parameter_group('test.ab.bc.cd')
        traj.cd.add_link(traj.test)
        traj.test.add_link(traj.cd)

        traj.store()

        traj2 = Trajectory(filename=filename)
        traj2.load(name=traj.name, load_data=2)

        self.assertTrue(traj.kk == traj2.gg, '%s != %s' % (traj.kk, traj2.gg))
        self.assertTrue(traj.cd.test is traj.test)

        self.assertTrue(len(traj._linked_by), len(traj2._linked_by))
        self.compare_trajectories(traj, traj2)

        self.assertTrue('jj' in traj2._nn_interface._links_count)
        traj2.remove_child('jj')
        self.assertTrue('jj' not in traj2._nn_interface._links_count)
        traj2.remove_child('hh')
        traj2.remove_child('ii')



        traj2.remove_child('parameters', recursive=True)

        traj2.auto_load = True

        group = traj2.par.test2.circle2

        self.assertTrue(group is traj2.test)

        retest = traj2.test.circle1

        self.assertTrue(retest is traj2.test2)

        self.assertTrue(traj2.test.circle2 is traj2.test)

        self.assertTrue(traj2.hh == traj2.res.kk)

        traj2.auto_load = False
        traj2.load_child('jj')
        self.assertTrue(traj2.jj is traj2.par)
        traj2.load(load_data=2)
        self.assertTrue(traj2.ii == traj2.res.kk)

    def test_find_in_all_runs_with_links(self):

        traj = Trajectory()

        traj.add_parameter('FloatParam')
        traj.par.FloatParam=4.0
        self.explore_dict = {'FloatParam':[1.0,1.1,1.2,1.3]}
        traj.explore(self.explore_dict)

        self.assertTrue(len(traj) == 4)

        traj.add_result('results.runs.run_00000000.sub.resulttest', 42)
        traj.add_result('results.runs.run_00000001.sub.resulttest', 43)
        traj.add_result('results.runs.run_00000002.sub.resulttest', 44)

        traj.add_result('results.runs.run_00000002.sub.resulttest2', 42)
        traj.add_result('results.runs.run_00000003.sub.resulttest2', 43)

        traj.add_derived_parameter('derived_parameters.runs.run_00000002.testing', 44)

        res_dict = traj.get_from_runs('resulttest', fast_access=True)

        self.assertTrue(len(res_dict)==3)
        self.assertTrue(res_dict['run_00000001']==43)
        self.assertTrue('run_00000003' not in res_dict)

        res_dict = traj.get_from_runs(name='sub.resulttest2', use_indices=True)

        self.assertTrue(len(res_dict)==2)
        self.assertTrue(res_dict[3] is traj.get('run_00000003.resulttest2'))
        self.assertTrue(1 not in res_dict)

        traj.res.runs.r_0.add_link('resulttest2', traj.r_1.get('resulttest'))

        res_dict = traj.get_from_runs(name='resulttest2', use_indices=True)

        self.assertTrue(len(res_dict)==3)
        self.assertTrue(res_dict[0] is traj.get('run_00000001.resulttest'))
        self.assertTrue(1 not in res_dict)

        res_dict = traj.get_from_runs(name='resulttest2', use_indices=True, with_links=False)

        self.assertTrue(len(res_dict)==2)
        self.assertTrue(0 not in res_dict)
        self.assertTrue(1 not in res_dict)

    def test_get_all_not_links(self):

        traj = Trajectory()

        traj.add_parameter('test.hi', 44)
        traj.explore({'hi': [1,2,3]})

        traj.add_parameter_group('test.test.test2')
        traj.add_parameter_group('test2')
        traj.test2.add_link('test', traj.test)

        nodes = traj.get_all('par.test')

        self.assertTrue(len(nodes) == 2)

        nodes = traj.get_all('par.test', shortcuts=False)

        self.assertTrue(len(nodes) == 1)

        traj.set_crun(0)

        traj.add_group('f.$.h')
        traj.add_group('f.$.g.h')
        traj.add_group('f.$.i')
        traj.crun.i.add_link('h', traj.crun.h)

        nodes = traj.get_all('$.h')

        self.assertTrue(len(nodes)==2)

        nodes = traj.get_all('h')

        self.assertTrue(len(nodes)==2)

        traj.idx = -1

        nodes = traj.get_all('h')

        self.assertTrue(len(nodes)==2)

    def test_links_according_to_run(self):

        traj = Trajectory()

        traj.add_parameter('test.hi', 44)
        traj.explore({'hi': [1,2,3]})

        traj.add_parameter_group('test.test.test2')
        traj.add_parameter_group('test2')
        traj.test2.add_link('test', traj.test)

        traj.idx = 1

    def test_link_deletion(self):
        filename = make_temp_dir('linktest2.hdf5')
        traj = Trajectory(filename=filename)

        traj.add_parameter_group('test')
        traj.add_parameter_group('test2')
        res= traj.add_result('kk', 42)
        traj.par.add_link('gg', res)

        traj.test.add_link('circle1' , traj.test2)
        traj.test2.add_link('circle2' , traj.test)

        traj.store()

        traj.delete_link('par.gg')

        traj2 = Trajectory(filename=filename)
        traj2.load(name=traj.name, load_data=2)

        with self.assertRaises(AttributeError):
            traj2.gg

if __name__ == '__main__':
    opt_args = parse_args()
    run_suite(**opt_args)