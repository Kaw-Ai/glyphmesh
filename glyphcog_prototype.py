#!/usr/bin/env python3
"""
GlyphCog Prototype Implementation
=================================

A minimal working prototype demonstrating the GlyphCog concept of extending 
the Pointwise Glyph framework for cognitive AI architectures.

This prototype implements basic geometric cognitive operations as a proof 
of concept for the full roadmap implementation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class DifferentialManifold(ABC):
    """Base class for differential manifolds"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        
    @abstractmethod
    def compute_metric_tensor(self, point: np.ndarray) -> np.ndarray:
        """Compute metric tensor at a point"""
        pass
        
    @abstractmethod
    def compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute geodesic distance between points"""
        pass
        
    def compute_geodesic(self, start: np.ndarray, end: np.ndarray, steps: int = 50) -> np.ndarray:
        """Simple linear interpolation as geodesic approximation"""
        # This is a simplified implementation - real geodesics require solving differential equations
        t = np.linspace(0, 1, steps)
        return np.array([start + s * (end - start) for s in t])


class HyperbolicManifold(DifferentialManifold):
    """Hyperbolic manifold for hierarchical cognitive structures"""
    
    def __init__(self, dimension: int, curvature: float = -1.0):
        super().__init__(dimension)
        self.curvature = curvature
        
    def compute_metric_tensor(self, point: np.ndarray) -> np.ndarray:
        """Hyperbolic metric tensor (Poincaré ball model)"""
        norm_squared = np.sum(point**2)
        if norm_squared >= 1.0:
            norm_squared = 0.99  # Stay within unit ball
        factor = 4 / (1 - norm_squared)**2
        return factor * np.eye(self.dimension)
        
    def compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Hyperbolic distance in Poincaré ball model"""
        # Ensure points are in unit ball
        point1 = point1 / (np.linalg.norm(point1) + 1e-8) * 0.99
        point2 = point2 / (np.linalg.norm(point2) + 1e-8) * 0.99
        
        norm1_sq = np.sum(point1**2)
        norm2_sq = np.sum(point2**2)
        diff_norm_sq = np.sum((point1 - point2)**2)
        
        numerator = 2 * diff_norm_sq
        denominator = (1 - norm1_sq) * (1 - norm2_sq)
        
        if denominator <= 0:
            return float('inf')
            
        return np.arccosh(1 + numerator / denominator)


class EuclideanManifold(DifferentialManifold):
    """Standard Euclidean space"""
    
    def compute_metric_tensor(self, point: np.ndarray) -> np.ndarray:
        """Euclidean metric tensor"""
        return np.eye(self.dimension)
        
    def compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Euclidean distance"""
        return np.linalg.norm(point1 - point2)


class GeometricGrammar:
    """Simple geometric grammar for cognitive operations"""
    
    def __init__(self):
        self.operations = {
            'attention': self._attention_operation,
            'memory': self._memory_operation,
            'transform': self._transform_operation
        }
        
    def parse_on_manifold(self, 
                         input_data: np.ndarray,
                         operation: str = 'attention',
                         **kwargs) -> np.ndarray:
        """Parse input using geometric operations"""
        if operation in self.operations:
            return self.operations[operation](input_data, **kwargs)
        else:
            return input_data
            
    def _attention_operation(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Implement attention as geometric focusing"""
        # Simple attention: amplify data based on distance from center
        center = kwargs.get('center', np.zeros(data.shape[-1]))
        distances = np.array([np.linalg.norm(point - center) for point in data])
        weights = np.exp(-distances)  # Closer points get higher weights
        weights = weights / np.sum(weights)  # Normalize
        return data * weights.reshape(-1, 1)
        
    def _memory_operation(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Simple memory operation"""
        # Memory as local averaging
        return np.mean(data, axis=0, keepdims=True)
        
    def _transform_operation(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Simple geometric transformation"""
        transform_matrix = kwargs.get('transform', np.eye(data.shape[-1]))
        return np.array([transform_matrix @ point for point in data])


class AgenticMicrokernel:
    """Autonomous processing unit for cognitive domains"""
    
    def __init__(self, kernel_type: str, grammar: GeometricGrammar):
        self.kernel_type = kernel_type
        self.grammar = grammar
        self.domain = None
        self.local_state = {}
        
    def bind_to_domain(self, domain):
        """Bind microkernel to a cognitive domain"""
        self.domain = domain
        
    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Process input data using geometric grammar"""
        # Simple processing pipeline
        embedded_input = self._embed_input(input_data)
        parsed_result = self.grammar.parse_on_manifold(
            embedded_input, 
            operation=self.kernel_type
        )
        return self._project_output(parsed_result)
        
    def _embed_input(self, input_data: np.ndarray) -> np.ndarray:
        """Embed input in manifold space"""
        # Simple embedding: normalize to unit sphere
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        norms = np.linalg.norm(input_data, axis=1, keepdims=True)
        return input_data / (norms + 1e-8)
        
    def _project_output(self, processed_data: np.ndarray) -> np.ndarray:
        """Project processed data back to output space"""
        return processed_data


class CognitiveDomain:
    """Cognitive domain supporting geometric operations"""
    
    def __init__(self, 
                 topology_type: str = "euclidean",
                 dimension: int = 3):
        
        self.manifold = self._create_manifold(topology_type, dimension)
        self.microkernels = []
        self.vector_fields = {}
        
    def _create_manifold(self, topology_type: str, dimension: int):
        """Create underlying manifold"""
        if topology_type == "hyperbolic":
            return HyperbolicManifold(dimension)
        else:
            return EuclideanManifold(dimension)
            
    def attach_microkernel(self, kernel: AgenticMicrokernel):
        """Attach microkernel to domain"""
        kernel.bind_to_domain(self)
        self.microkernels.append(kernel)
        
    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Process data through all attached microkernels"""
        result = input_data
        for kernel in self.microkernels:
            result = kernel.process(result)
        return result
        
    def compute_geodesic(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Compute geodesic path"""
        return self.manifold.compute_geodesic(start, end)


class HyperGlyph(CognitiveDomain):
    """Extended Glyph supporting multi-dimensional and multi-topology processing"""
    
    def __init__(self, 
                 primary_topology: str = "euclidean",
                 dimension: int = 3,
                 secondary_topologies: List[str] = None,
                 cross_dimensional_ops: bool = True):
        
        super().__init__(primary_topology, dimension)
        
        # Support for multiple concurrent topologies
        self.secondary_manifolds = {}
        if secondary_topologies:
            for i, topo in enumerate(secondary_topologies):
                manifold_id = f"secondary_{i}"
                self.secondary_manifolds[manifold_id] = self._create_manifold(topo, dimension)
        
        # Enhanced capabilities
        self.cross_dimensional_ops = cross_dimensional_ops
        self.dimension_adapters = {}
        self.geometric_transforms = []
        
    def add_dimensional_adapter(self, source_dim: int, target_dim: int):
        """Add adapter for cross-dimensional operations"""
        if self.cross_dimensional_ops:
            adapter_key = f"{source_dim}_to_{target_dim}"
            # Simple linear projection/embedding for dimension adaptation
            if source_dim < target_dim:
                # Embedding: pad with zeros
                self.dimension_adapters[adapter_key] = lambda x: np.pad(x, 
                    ((0, 0), (0, target_dim - source_dim)), mode='constant')
            else:
                # Projection: take first target_dim components
                self.dimension_adapters[adapter_key] = lambda x: x[:, :target_dim]
    
    def process_multi_topology(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process data through all available topologies"""
        results = {}
        
        # Process through primary manifold
        results['primary'] = self.process(input_data)
        
        # Process through secondary manifolds
        for manifold_id, manifold in self.secondary_manifolds.items():
            # Create temporary domain for processing
            temp_domain = CognitiveDomain(manifold.dimension)
            temp_domain.manifold = manifold
            
            # Copy microkernels to temp domain
            for kernel in self.microkernels:
                temp_domain.attach_microkernel(kernel)
            
            results[manifold_id] = temp_domain.process(input_data)
        
        return results
    
    def compute_cross_topology_distance(self, point1: np.ndarray, point2: np.ndarray) -> Dict[str, float]:
        """Compute distances in all available topologies"""
        distances = {}
        
        # Primary topology
        distances['primary'] = self.manifold.compute_distance(point1, point2)
        
        # Secondary topologies
        for manifold_id, manifold in self.secondary_manifolds.items():
            distances[manifold_id] = manifold.compute_distance(point1, point2)
        
        return distances
    
    def geometric_fusion(self, multi_results: Dict[str, np.ndarray], fusion_type: str = "weighted_average") -> np.ndarray:
        """Fuse results from multiple topologies"""
        if fusion_type == "weighted_average":
            # Simple weighted average (equal weights for now)
            all_results = list(multi_results.values())
            if len(all_results) == 1:
                return all_results[0]
            
            # Ensure all results have the same shape
            base_shape = all_results[0].shape
            valid_results = [r for r in all_results if r.shape == base_shape]
            
            if valid_results:
                return np.mean(valid_results, axis=0)
            else:
                return all_results[0]
        
        return multi_results['primary']  # Fallback to primary


class GlyphMorphicMesh:
    """Adaptive mesh structure analogous to Graph Neural Networks with geometric learning"""
    
    def __init__(self, 
                 initial_nodes: int = 10,
                 dimension: int = 3,
                 topology_type: str = "euclidean",
                 learning_rate: float = 0.01):
        
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.topology_type = topology_type
        
        # Initialize mesh structure
        self.nodes = self._initialize_nodes(initial_nodes)
        self.edges = self._initialize_edges()
        self.node_features = np.random.randn(initial_nodes, dimension) * 0.1
        
        # Learning components
        self.node_embeddings = {}
        self.edge_weights = {}
        self.adaptation_history = []
        
        # Geometric processing units
        self.manifold = self._create_manifold(topology_type, dimension)
        self.local_domains = {}
        
    def _create_manifold(self, topology_type: str, dimension: int):
        """Create underlying manifold for geometric operations"""
        if topology_type == "hyperbolic":
            return HyperbolicManifold(dimension)
        else:
            return EuclideanManifold(dimension)
    
    def _initialize_nodes(self, count: int) -> List[int]:
        """Initialize node indices"""
        return list(range(count))
    
    def _initialize_edges(self) -> Dict[Tuple[int, int], float]:
        """Initialize edges with geometric-based connectivity"""
        edges = {}
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    # Initialize edge weight based on random geometric distance
                    distance = np.random.exponential(0.5)  # Exponential decay for sparsity
                    if distance < 1.0:  # Only connect nearby nodes
                        edges[(i, j)] = 1.0 / (1.0 + distance)
        return edges
    
    def add_node(self, features: np.ndarray = None) -> int:
        """Dynamically add a new node to the mesh"""
        new_id = len(self.nodes)
        self.nodes.append(new_id)
        
        # Add features
        if features is None:
            features = np.random.randn(self.dimension) * 0.1
        
        # Expand node features array
        self.node_features = np.vstack([self.node_features, features.reshape(1, -1)])
        
        # Add edges to nearby nodes based on geometric proximity
        for existing_node in self.nodes[:-1]:
            distance = self.manifold.compute_distance(
                features, 
                self.node_features[existing_node]
            )
            if distance < 0.5:  # Threshold for connection
                weight = 1.0 / (1.0 + distance)
                self.edges[(existing_node, new_id)] = weight
                self.edges[(new_id, existing_node)] = weight
        
        return new_id
    
    def remove_node(self, node_id: int):
        """Remove a node and its connections"""
        if node_id in self.nodes:
            # Remove edges
            edges_to_remove = [(i, j) for (i, j) in self.edges.keys() 
                             if i == node_id or j == node_id]
            for edge in edges_to_remove:
                del self.edges[edge]
            
            # Remove from nodes list
            self.nodes.remove(node_id)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighboring nodes"""
        neighbors = []
        for (i, j) in self.edges.keys():
            if i == node_id:
                neighbors.append(j)
            elif j == node_id:
                neighbors.append(i)
        return list(set(neighbors))
    
    def message_passing_step(self, input_features: np.ndarray = None) -> np.ndarray:
        """Perform one step of geometric message passing (like GNN)"""
        if input_features is None:
            input_features = self.node_features
        
        new_features = np.copy(input_features)
        
        for node in self.nodes:
            if node >= len(input_features):
                continue
                
            neighbors = self.get_neighbors(node)
            if not neighbors:
                continue
            
            # Aggregate neighbor features weighted by edge weights and geometric distances
            aggregated = np.zeros(self.dimension)
            total_weight = 0.0
            
            for neighbor in neighbors:
                if neighbor < len(input_features):
                    edge_weight = self.edges.get((node, neighbor), 0.0)
                    if edge_weight == 0.0:
                        edge_weight = self.edges.get((neighbor, node), 0.0)
                    
                    # Geometric weighting based on manifold distance
                    geom_distance = self.manifold.compute_distance(
                        input_features[node], 
                        input_features[neighbor]
                    )
                    geom_weight = 1.0 / (1.0 + geom_distance)
                    
                    combined_weight = edge_weight * geom_weight
                    aggregated += combined_weight * input_features[neighbor]
                    total_weight += combined_weight
            
            if total_weight > 0:
                # Update with learning rate
                aggregated = aggregated / total_weight
                new_features[node] = (1 - self.learning_rate) * input_features[node] + \
                                   self.learning_rate * aggregated
        
        return new_features
    
    def adapt_topology(self, input_data: np.ndarray, performance_metric: float = None):
        """Adapt mesh topology based on processing performance"""
        # Simple adaptation: add nodes where gradients are high
        if performance_metric is None:
            # Use variance as a simple performance metric
            performance_metric = np.var(input_data)
        
        # Record adaptation
        self.adaptation_history.append({
            'step': len(self.adaptation_history),
            'performance': performance_metric,
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges)
        })
        
        # Adaptive rules
        if performance_metric > 0.5:  # High variance suggests need for more complexity
            # Add a node at the centroid of high-variance region
            centroid = np.mean(input_data, axis=0)
            self.add_node(centroid)
        
        elif performance_metric < 0.1 and len(self.nodes) > 5:  # Low variance, maybe oversimplified
            # Remove least connected node
            node_connections = {node: len(self.get_neighbors(node)) for node in self.nodes}
            least_connected = min(node_connections, key=node_connections.get)
            if node_connections[least_connected] <= 1:
                self.remove_node(least_connected)
    
    def process_sequence(self, input_sequence: List[np.ndarray], steps: int = 3) -> List[np.ndarray]:
        """Process a sequence of inputs with learning and adaptation"""
        results = []
        
        for input_data in input_sequence:
            # Multiple message passing steps
            current_features = self.node_features
            for _ in range(steps):
                current_features = self.message_passing_step(current_features)
            
            # Update node features
            self.node_features = current_features
            
            # Adapt topology based on processing
            performance = np.var(current_features)
            self.adapt_topology(input_data, performance)
            
            results.append(current_features)
        
        return results
    
    def get_mesh_summary(self) -> Dict:
        """Get summary of current mesh state"""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'dimension': self.dimension,
            'topology_type': self.topology_type,
            'adaptation_steps': len(self.adaptation_history)
        }


class HierarchicalCognitiveArchitecture:
    """Hierarchical cognitive architecture using multiple domains"""
    
    def __init__(self):
        self.levels = []
        self.connections = []
        
    def add_level(self, domain: CognitiveDomain, level_name: str):
        """Add a cognitive level"""
        self.levels.append({
            'name': level_name,
            'domain': domain
        })
        
    def add_connection(self, source_level: int, target_level: int, connection_type: str = "feed_forward"):
        """Add connection between levels"""
        self.connections.append({
            'source': source_level,
            'target': target_level,
            'type': connection_type
        })
        
    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Process data through hierarchical architecture"""
        current_data = input_data
        
        # Process through each level
        for level in self.levels:
            current_data = level['domain'].process(current_data)
            
        return current_data


class CognitiveGlyphAPI:
    """Simplified API for cognitive operations"""
    
    def __init__(self):
        self.domains = {}
        self.architectures = {}
        self.hyper_glyphs = {}
        self.morphic_meshes = {}
        
    def create_cognitive_domain(self, 
                              domain_id: str,
                              topology_type: str = "euclidean",
                              dimension: int = 3) -> CognitiveDomain:
        """Create a new cognitive domain"""
        domain = CognitiveDomain(topology_type, dimension)
        self.domains[domain_id] = domain
        return domain
    
    def create_hyper_glyph(self,
                          glyph_id: str,
                          primary_topology: str = "euclidean",
                          dimension: int = 3,
                          secondary_topologies: List[str] = None) -> HyperGlyph:
        """Create a new HyperGlyph with multi-topology support"""
        hyper_glyph = HyperGlyph(
            primary_topology=primary_topology,
            dimension=dimension,
            secondary_topologies=secondary_topologies or []
        )
        self.hyper_glyphs[glyph_id] = hyper_glyph
        return hyper_glyph
    
    def create_morphic_mesh(self,
                           mesh_id: str,
                           initial_nodes: int = 10,
                           dimension: int = 3,
                           topology_type: str = "euclidean") -> GlyphMorphicMesh:
        """Create a new adaptive GlyphMorphicMesh"""
        morphic_mesh = GlyphMorphicMesh(
            initial_nodes=initial_nodes,
            dimension=dimension,
            topology_type=topology_type
        )
        self.morphic_meshes[mesh_id] = morphic_mesh
        return morphic_mesh
        
    def create_attention_architecture(self, 
                                    arch_id: str,
                                    input_dimension: int = 10,
                                    hidden_dimension: int = 5) -> HierarchicalCognitiveArchitecture:
        """Create a simple attention-based architecture"""
        
        # Create domains for different processing stages
        input_domain = self.create_cognitive_domain(
            f"{arch_id}_input", "euclidean", input_dimension
        )
        attention_domain = self.create_cognitive_domain(
            f"{arch_id}_attention", "hyperbolic", hidden_dimension
        )
        output_domain = self.create_cognitive_domain(
            f"{arch_id}_output", "euclidean", input_dimension
        )
        
        # Create microkernels
        grammar = GeometricGrammar()
        input_kernel = AgenticMicrokernel("transform", grammar)
        attention_kernel = AgenticMicrokernel("attention", grammar)
        output_kernel = AgenticMicrokernel("memory", grammar)
        
        # Attach microkernels to domains
        input_domain.attach_microkernel(input_kernel)
        attention_domain.attach_microkernel(attention_kernel)
        output_domain.attach_microkernel(output_kernel)
        
        # Create hierarchical architecture
        architecture = HierarchicalCognitiveArchitecture()
        architecture.add_level(input_domain, "input")
        architecture.add_level(attention_domain, "attention")
        architecture.add_level(output_domain, "output")
        
        architecture.add_connection(0, 1, "feed_forward")
        architecture.add_connection(1, 2, "feed_forward")
        
        self.architectures[arch_id] = architecture
        return architecture


def demo_basic_operations():
    """Demonstrate basic GlyphCog operations"""
    print("GlyphCog Prototype Demo")
    print("======================\n")
    
    # Create API instance
    api = CognitiveGlyphAPI()
    
    # Test 1: Basic manifold operations
    print("1. Testing Manifold Operations:")
    euclidean_domain = api.create_cognitive_domain("test_euclidean", "euclidean", 2)
    hyperbolic_domain = api.create_cognitive_domain("test_hyperbolic", "hyperbolic", 2)
    
    point1 = np.array([0.1, 0.2])
    point2 = np.array([0.3, 0.4])
    
    euclidean_distance = euclidean_domain.manifold.compute_distance(point1, point2)
    hyperbolic_distance = hyperbolic_domain.manifold.compute_distance(point1, point2)
    
    print(f"Euclidean distance: {euclidean_distance:.4f}")
    print(f"Hyperbolic distance: {hyperbolic_distance:.4f}")
    
    # Test 2: Microkernel processing
    print("\n2. Testing Microkernel Processing:")
    grammar = GeometricGrammar()
    attention_kernel = AgenticMicrokernel("attention", grammar)
    euclidean_domain.attach_microkernel(attention_kernel)
    
    # Create test data
    test_data = np.random.randn(5, 2) * 0.1  # Small random points
    print(f"Input data shape: {test_data.shape}")
    
    # Process through domain
    processed_data = euclidean_domain.process(test_data)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processing changed data: {not np.allclose(test_data, processed_data)}")
    
    # Test 3: Hierarchical architecture
    print("\n3. Testing Hierarchical Architecture:")
    architecture = api.create_attention_architecture("demo_arch", input_dimension=4, hidden_dimension=3)
    
    # Test input
    test_input = np.random.randn(3, 4) * 0.1
    print(f"Architecture input shape: {test_input.shape}")
    
    # Process through architecture
    output = architecture.process(test_input)
    print(f"Architecture output shape: {output.shape}")
    
    print("\nDemo completed successfully!")


def demo_hyper_glyph():
    """Demonstrate HyperGlyph multi-topology processing"""
    print("\n" + "="*50)
    print("HyperGlyph Multi-Topology Demo")
    print("="*50)
    
    api = CognitiveGlyphAPI()
    
    # Create HyperGlyph with multiple topologies
    hyper_glyph = api.create_hyper_glyph(
        "demo_hyper",
        primary_topology="euclidean",
        dimension=3,
        secondary_topologies=["hyperbolic"]
    )
    
    # Add some microkernels
    grammar = GeometricGrammar()
    attention_kernel = AgenticMicrokernel("attention", grammar)
    transform_kernel = AgenticMicrokernel("transform", grammar)
    
    hyper_glyph.attach_microkernel(attention_kernel)
    hyper_glyph.attach_microkernel(transform_kernel)
    
    # Test data
    test_points = np.random.randn(4, 3) * 0.1
    print(f"Input data shape: {test_points.shape}")
    
    # Process through multiple topologies
    multi_results = hyper_glyph.process_multi_topology(test_points)
    print(f"Processing topologies: {list(multi_results.keys())}")
    for topology, result in multi_results.items():
        print(f"  {topology}: shape {result.shape}")
    
    # Test cross-topology distances
    point1, point2 = test_points[0], test_points[1]
    distances = hyper_glyph.compute_cross_topology_distance(point1, point2)
    print(f"\nCross-topology distances:")
    for topology, distance in distances.items():
        print(f"  {topology}: {distance:.4f}")
    
    # Test geometric fusion
    fused_result = hyper_glyph.geometric_fusion(multi_results)
    print(f"\nFused result shape: {fused_result.shape}")
    
    print("HyperGlyph demo completed!")


def demo_glyph_morphic_mesh():
    """Demonstrate GlyphMorphicMesh adaptive behavior"""
    print("\n" + "="*50)
    print("GlyphMorphicMesh Adaptive Demo")
    print("="*50)
    
    api = CognitiveGlyphAPI()
    
    # Create adaptive mesh
    morphic_mesh = api.create_morphic_mesh(
        "demo_mesh",
        initial_nodes=8,
        dimension=3,
        topology_type="euclidean"
    )
    
    print(f"Initial mesh: {morphic_mesh.get_mesh_summary()}")
    
    # Create a sequence of inputs with varying complexity
    input_sequence = [
        np.random.randn(3, 3) * 0.1,  # Low variance
        np.random.randn(3, 3) * 0.5,  # Medium variance  
        np.random.randn(3, 3) * 1.0,  # High variance
        np.random.randn(3, 3) * 0.2,  # Low variance again
    ]
    
    print(f"\nProcessing sequence of {len(input_sequence)} inputs...")
    
    # Process sequence and observe adaptation
    results = morphic_mesh.process_sequence(input_sequence, steps=2)
    
    print(f"Final mesh: {morphic_mesh.get_mesh_summary()}")
    
    # Show adaptation history
    print("\nAdaptation History:")
    for i, adaptation in enumerate(morphic_mesh.adaptation_history):
        print(f"  Step {i}: nodes={adaptation['num_nodes']}, "
              f"edges={adaptation['num_edges']}, "
              f"performance={adaptation['performance']:.3f}")
    
    # Test message passing
    print(f"\nFinal processing result shape: {results[-1].shape}")
    print("GlyphMorphicMesh demo completed!")


def demo_novel_geometric_forms():
    """Comprehensive demo of all novel geometric forms"""
    print("Novel Geometric Forms Demo")
    print("=" * 50)
    
    # Demo basic Glyph (existing)
    demo_basic_operations()
    
    # Demo HyperGlyph 
    demo_hyper_glyph()
    
    # Demo GlyphMorphicMesh
    demo_glyph_morphic_mesh()
    
    print("\n" + "="*50)
    print("All Novel Geometric Forms Demonstrated Successfully!")
    print("="*50)


def visualize_geodesics():
    """Visualize geodesics on different manifolds"""
    print("\n4. Visualizing Geodesics:")
    
    # Create manifolds
    euclidean = EuclideanManifold(2)
    hyperbolic = HyperbolicManifold(2)
    
    # Define points
    start = np.array([0.1, 0.1])
    end = np.array([0.6, 0.4])
    
    # Compute geodesics
    euclidean_geodesic = euclidean.compute_geodesic(start, end, 50)
    hyperbolic_geodesic = hyperbolic.compute_geodesic(start, end, 50)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(euclidean_geodesic[:, 0], euclidean_geodesic[:, 1], 'b-', linewidth=2, label='Euclidean Geodesic')
    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(end[0], end[1], 'ro', markersize=8, label='End')
    plt.title('Euclidean Manifold')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(hyperbolic_geodesic[:, 0], hyperbolic_geodesic[:, 1], 'r-', linewidth=2, label='Hyperbolic Geodesic')
    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(end[0], end[1], 'ro', markersize=8, label='End')
    
    # Draw unit circle for Poincaré ball
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    plt.plot(circle_x, circle_y, 'k--', alpha=0.3, label='Unit Circle')
    
    plt.title('Hyperbolic Manifold (Poincaré Ball)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/glyphmesh/glyphmesh/docs/geodesics_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Geodesic visualization saved to docs/geodesics_demo.png")


if __name__ == "__main__":
    # Run the comprehensive demo of novel geometric forms
    demo_novel_geometric_forms()
    
    # Create visualization
    try:
        visualize_geodesics()
    except Exception as e:
        print(f"Visualization failed (likely no display): {e}")
        print("Run with display for geodesic visualization")
    
    print("\nGlyphCog prototype demonstration completed!")
    print("\nImplemented Novel Geometric Forms:")
    print("- Glyph: Base cognitive domain with geometric processing")
    print("- HyperGlyph: Multi-topology processing with cross-dimensional operations")
    print("- GlyphMorphicMesh: Adaptive mesh analogous to Graph Neural Networks")
    print("\nNext steps:")
    print("- Implement full differential geometry operations")
    print("- Add proper Pointwise integration")
    print("- Develop advanced learning algorithms")
    print("- Create comprehensive test suite")
    print("- Build visualization and debugging tools")