from gfn.containers.replay_buffer import ReplayBuffer
from utils import RandomProjectionEncoder
class TopologicalExperienceReplayBuffer(ReplayBuffer):
    """Topological Experience Replay Buffer.

    This class implements a topological experience replay buffer, which is a
    replay buffer that stores transitions in a topological order. This buffer
    is useful for training graph-based models, as it ensures that the transitions
    are stored in a way that respects the graph structure.
    It uses reverse-sweep to sample terminal vertices and predecessors.

    Parameters
    ----------
    capacity : int
        The maximum number of transitions that the buffer can store.
    """

    def __init__(self, capacity: int, t_warm_up: int, num_sampled_terminal_vertices: int = 8, num_sampled_predecessors: int = 3) -> None:
        super().__init__(capacity)
        # topology, which stores the graph structure
        self.graph = {}
        # warmup period, during which transitions are stitched together to form the graph
        self.t_warm_up = t_warm_up
        # search queue, which stores the sampled terminal vertices.
        self.search_queue = []
        # batch queue, which stores the batch of transitions         
        self.batch_queue = []
        # number of sampled terminal vertices during reverse sweep
        self.num_sampled_terminal_vertices = num_sampled_terminal_vertices
        # number of sampled predecessors from each terminal vertex
        self.num_sampled_predecessors = num_sampled_predecessors
        # encoder, which encodes the nodes of the graph
        self.node_encoder = RandomProjectionEncoder(input_shape=env.state_shape)
        
    def set_topology(self, topology: Container) -> None:
        """Sets the topology of the graph.

        Parameters
        ----------
        topology : Container
            The topology of the graph.
        """
        self.topology = topology

    def sample(self, n_samples: int) -> Container:
        """Samples a subset of the buffer.

        Parameters
        ----------
        n_samples : int
            The number of samples to return.

        Returns
        -------
        Container
            A container containing the sampled transitions.
        """
        if self.topology is None:
            raise ValueError("Topology not set.")
        return self[self.topology.sample(n_samples)]