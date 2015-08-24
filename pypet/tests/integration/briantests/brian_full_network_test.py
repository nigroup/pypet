__author__ = 'Robert Meyer'

from pypet.tests.testutils.ioutils import make_temp_dir, run_suite,  \
    get_root_logger, parse_args, get_log_config
from pypet.tests.testutils.ioutils import unittest

try:
    import brian
    from brian import *
    from pypet.brian.parameter import BrianParameter, BrianMonitorResult
except ImportError as exc:
    #print('Import Error: %s' % str(exc))
    brian = None

from pypet.tests.testutils.data import TrajectoryComparator
from pypet.trajectory import Trajectory
from pypet.environment import Environment

import logging
from pypet.utils.explore import cartesian_product
import time
import os



def add_params(traj):

    traj.standard_parameter=BrianParameter
    traj.fast_access=True

    traj.add_parameter('Sim.defaultclock', 0.01*ms)
    traj.add_parameter('Net.C',281*pF)
    traj.add_parameter('Net.gL',30*nS)
    traj.add_parameter('Net.EL',-70.6*mV)
    traj.add_parameter('Net.VT',-50.4*mV)
    traj.add_parameter('Net.DeltaT',2*mV)
    traj.add_parameter('Net.tauw',40*ms)
    traj.add_parameter('Net.a',4*nS)
    traj.add_parameter('Net.b',0.08*nA)
    traj.add_parameter('Net.I',.8*nA)
    traj.add_parameter('Net.Vcut',traj.VT+5*traj.DeltaT) # practical threshold condition
    traj.add_parameter('Net.N',100)

    eqs="""
    dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I-w)/C : volt
    dw/dt=(a*(vm-EL)-w)/tauw : amp
    Vr:volt
    """

    traj.add_parameter('Net.eqs', eqs)
    traj.add_parameter('reset', 'vm=Vr;w+=b')
    pass

def run_net(traj):

    clear(True, True)
    get_root_logger().info(traj.defaultclock)
    defaultclock.dt=traj.defaultclock

    C=traj.C
    gL=traj.gL
    EL=traj.EL
    VT=traj.VT
    DeltaT=traj.DeltaT
    tauw=traj.tauw
    a=traj.a
    b=traj.b
    I=traj.I
    Vcut=traj.Vcut# practical threshold condition
    N=traj.N

    eqs=traj.eqs

    neuron=NeuronGroup(N,model=eqs,threshold=Vcut,reset=traj.reset)
    neuron.vm=EL
    neuron.w=a*(neuron.vm-EL)
    neuron.Vr=linspace(-48.3*mV,-47.7*mV,N) # bifurcation parameter

    #run(25*msecond,report='text') # we discard the first spikes

    MSpike=SpikeMonitor(neuron, delay = 1*ms) # record Vr and w at spike times
    MPopSpike =PopulationSpikeCounter(neuron, delay = 1*ms)
    MPopRate = PopulationRateMonitor(neuron,bin=5*ms)
    MStateV = StateMonitor(neuron,'vm',record=[1,2,3])
    MStatewMean = StateMonitor(neuron,'w',record=False)

    MRecentStateV = RecentStateMonitor(neuron,'vm',record=[1,2,3],duration=10*ms)
    MRecentStatewMean = RecentStateMonitor(neuron,'w',duration=10*ms,record=False)

    MCounts = SpikeCounter(neuron)

    MStateSpike = StateSpikeMonitor(neuron,('w','vm'))

    MMultiState = MultiStateMonitor(neuron,['w','vm'],record=[6,7,8,9])

    ISIHist = ISIHistogramMonitor(neuron,[0,0.0001,0.0002], delay = 1*ms)

    VanRossum = VanRossumMetric(neuron, tau=5*msecond)

    run(25*msecond,report='text')

    traj.standard_result = BrianMonitorResult

    traj.add_result('SpikeMonitor', MSpike)
    traj.add_result('SpikeMonitorAr', MSpike, storage_mode = BrianMonitorResult.ARRAY_MODE)
    traj.add_result('PopulationSpikeCounter', MPopSpike)
    traj.add_result('PopulationRateMonitor',MPopRate)
    traj.add_result('StateMonitorV', MStateV)
    traj.add_result('StateMonitorwMean', MStatewMean)
    traj.add_result('Counts',MCounts)

    traj.add_result('StateSpikevmw', MStateSpike)
    traj.add_result('StateSpikevmwAr', MStateSpike,storage_mode = BrianMonitorResult.ARRAY_MODE)
    traj.add_result('MultiState',MMultiState)
    traj.add_result('ISIHistogrammMonitor',ISIHist)
    traj.add_result('RecentStateMonitorV', MRecentStateV)
    traj.add_result('RecentStateMonitorwMean', MRecentStatewMean)
    traj.add_result('VanRossumMetric', VanRossum)


@unittest.skipIf(brian is None, 'Can only be run with brian!')
class BrianFullNetworkTest(TrajectoryComparator):

    tags = 'brian', 'integration'  # Test tags

    def tearDown(self):
        self.env.disable_logging()
        super(BrianFullNetworkTest, self).tearDown()

    def setUp(self):
        env = Environment(trajectory='Test_'+repr(time.time()).replace('.','_'),
                          filename=make_temp_dir(os.path.join(
                              'experiments',
                              'tests',
                              'briantests',
                              'HDF5',
                               'briantest.hdf5')),
                          file_title='test',
                          log_config=get_log_config(),
                          dynamic_imports=['pypet.brian.parameter.BrianParameter',
                                                        BrianMonitorResult],
                          multiproc=False)

        traj = env.trajectory

        #env._set_standard_storage()
        #env._hdf5_queue_writer._hdf5storageservice = LazyStorageService()
        traj = env.trajectory
        #traj.set_storage_service(LazyStorageService())

        add_params(traj)
        #traj.mode='Parallel'


        traj.explore(cartesian_product({traj.get('N').full_name:[50,60],
                               traj.get('tauw').full_name:[30*ms,40*ms]}))

        self.traj = traj

        self.env = env
        self.traj = traj


    def test_net(self):
        self.env.run(run_net)

        self.traj.load(load_derived_parameters=2, load_results=2)

        traj2 = Trajectory(name = self.traj.name, add_time=False,
                           filename=make_temp_dir(os.path.join(
                               'experiments',
                               'tests',
                               'briantests',
                               'HDF5',
                               'briantest.hdf5')),
                           dynamic_imports=['pypet.brian.parameter.BrianParameter',
                                                        BrianMonitorResult])

        traj2.load(load_parameters=2, load_derived_parameters=2, load_results=2)

        self.compare_trajectories(self.traj, traj2)


@unittest.skipIf(brian is None, 'Can only be run with brian!')
class BrianFullNetworkMPTest(BrianFullNetworkTest):

    tags = 'brian', 'multiproc', 'integration'  # Test tags

    def setUp(self):
        logging.basicConfig(level = logging.ERROR)


        env = Environment(trajectory='Test_'+repr(time.time()).replace('.','_'),
                          filename=make_temp_dir(os.path.join(
                              'experiments',
                              'tests',
                              'briantests',
                              'HDF5',
                              'briantest.hdf5')),
                          file_title='test',
                          log_config=get_log_config(),
                          dynamic_imports=['pypet.brian.parameter.BrianParameter',
                                                        BrianMonitorResult],
                          multiproc=True,
                          use_pool=True,
                          complib='blosc',
                          wrap_mode='QUEUE',
                          ncores=2)

        traj = env.trajectory

        #env._set_standard_storage()
        #env._hdf5_queue_writer._hdf5storageservice = LazyStorageService()
        traj = env.trajectory
        #traj.set_storage_service(LazyStorageService())

        add_params(traj)
        #traj.mode='Parallel'


        traj.explore(cartesian_product({traj.get('N').full_name:[50,60],
                               traj.get('tauw').full_name:[30*ms,40*ms]}))

        self.traj = traj

        self.env = env
        self.traj = traj


if __name__ == '__main__':
    opt_args = parse_args()
    run_suite(**opt_args)