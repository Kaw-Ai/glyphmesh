#!/usr/bin/env python3
"""
Test Suite for Novel Geometric Forms in GlyphCog
================================================

Tests for the newly implemented HyperGlyph and GlyphMorphicMesh classes.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import the prototype
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glyphcog_prototype import (
    CognitiveGlyphAPI, HyperGlyph, GlyphMorphicMesh, 
    GeometricGrammar, AgenticMicrokernel
)


def test_hyper_glyph_creation():
    """Test HyperGlyph creation and basic functionality"""
    print("Testing HyperGlyph creation...")
    
    api = CognitiveGlyphAPI()
    
    # Test basic creation
    hyper_glyph = api.create_hyper_glyph("test_hyper")
    assert hyper_glyph is not None
    assert len(hyper_glyph.secondary_manifolds) == 0
    
    # Test with secondary topologies
    hyper_glyph2 = api.create_hyper_glyph(
        "test_hyper2",
        secondary_topologies=["hyperbolic"]
    )
    assert len(hyper_glyph2.secondary_manifolds) == 1
    
    print("✓ HyperGlyph creation tests passed")


def test_hyper_glyph_multi_topology():
    """Test HyperGlyph multi-topology processing"""
    print("Testing HyperGlyph multi-topology processing...")
    
    api = CognitiveGlyphAPI()
    hyper_glyph = api.create_hyper_glyph(
        "test_multi",
        primary_topology="euclidean",
        secondary_topologies=["hyperbolic"]
    )
    
    # Add a microkernel
    grammar = GeometricGrammar()
    kernel = AgenticMicrokernel("attention", grammar)
    hyper_glyph.attach_microkernel(kernel)
    
    # Test data
    test_data = np.random.randn(3, 3) * 0.1
    
    # Multi-topology processing
    results = hyper_glyph.process_multi_topology(test_data)
    assert 'primary' in results
    assert 'secondary_0' in results
    assert results['primary'].shape == test_data.shape
    
    # Cross-topology distances
    distances = hyper_glyph.compute_cross_topology_distance(
        test_data[0], test_data[1]
    )
    assert 'primary' in distances
    assert 'secondary_0' in distances
    assert distances['primary'] >= 0
    assert distances['secondary_0'] >= 0
    
    # Geometric fusion
    fused = hyper_glyph.geometric_fusion(results)
    assert fused.shape == test_data.shape
    
    print("✓ HyperGlyph multi-topology tests passed")


def test_glyph_morphic_mesh_creation():
    """Test GlyphMorphicMesh creation and basic properties"""
    print("Testing GlyphMorphicMesh creation...")
    
    api = CognitiveGlyphAPI()
    
    # Basic creation
    mesh = api.create_morphic_mesh("test_mesh")
    assert mesh is not None
    assert len(mesh.nodes) == 10  # Default initial nodes
    assert mesh.dimension == 3  # Default dimension
    
    # Custom creation
    mesh2 = api.create_morphic_mesh(
        "test_mesh2",
        initial_nodes=5,
        dimension=2,
        topology_type="hyperbolic"
    )
    assert len(mesh2.nodes) == 5
    assert mesh2.dimension == 2
    assert mesh2.topology_type == "hyperbolic"
    
    print("✓ GlyphMorphicMesh creation tests passed")


def test_glyph_morphic_mesh_adaptation():
    """Test GlyphMorphicMesh adaptive behavior"""
    print("Testing GlyphMorphicMesh adaptation...")
    
    api = CognitiveGlyphAPI()
    mesh = api.create_morphic_mesh("test_adaptive", initial_nodes=5)
    
    initial_summary = mesh.get_mesh_summary()
    
    # Test node addition
    initial_node_count = len(mesh.nodes)
    new_features = np.random.randn(mesh.dimension) * 0.1
    new_node = mesh.add_node(new_features)
    assert len(mesh.nodes) == initial_node_count + 1
    assert new_node == initial_node_count
    
    # Test node removal
    mesh.remove_node(new_node)
    assert len(mesh.nodes) == initial_node_count
    
    # Test message passing
    test_features = np.random.randn(len(mesh.nodes), mesh.dimension) * 0.1
    mesh.node_features = test_features
    
    new_features = mesh.message_passing_step()
    assert new_features.shape == test_features.shape
    
    # Test sequence processing with adaptation
    input_sequence = [
        np.random.randn(3, mesh.dimension) * 0.1,
        np.random.randn(3, mesh.dimension) * 0.8,  # High variance
    ]
    
    results = mesh.process_sequence(input_sequence, steps=1)
    assert len(results) == len(input_sequence)
    assert len(mesh.adaptation_history) == len(input_sequence)
    
    print("✓ GlyphMorphicMesh adaptation tests passed")


def test_integration():
    """Test integration between different geometric forms"""
    print("Testing integration between geometric forms...")
    
    api = CognitiveGlyphAPI()
    
    # Create basic cognitive domain
    domain = api.create_cognitive_domain("basic", "euclidean", 3)
    
    # Create HyperGlyph
    hyper_glyph = api.create_hyper_glyph(
        "hyper", 
        secondary_topologies=["hyperbolic"]
    )
    
    # Create adaptive mesh
    mesh = api.create_morphic_mesh("mesh", initial_nodes=6, dimension=3)
    
    # Test that they can process the same data
    test_data = np.random.randn(4, 3) * 0.1
    
    # Process through basic domain
    basic_result = domain.process(test_data)
    
    # Process through HyperGlyph
    hyper_results = hyper_glyph.process_multi_topology(test_data)
    
    # Process through adaptive mesh
    mesh_results = mesh.process_sequence([test_data], steps=1)
    
    # Verify all produced valid outputs
    assert basic_result.shape == test_data.shape
    assert hyper_results['primary'].shape == test_data.shape
    assert len(mesh_results) == 1
    
    print("✓ Integration tests passed")


def run_all_tests():
    """Run all tests for novel geometric forms"""
    print("Running Tests for Novel Geometric Forms")
    print("=" * 50)
    
    try:
        test_hyper_glyph_creation()
        test_hyper_glyph_multi_topology()
        test_glyph_morphic_mesh_creation()
        test_glyph_morphic_mesh_adaptation()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("Novel geometric forms are working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)