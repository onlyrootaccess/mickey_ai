# Relationship mapping
"""
Knowledge Graph Module for Mickey AI
Stores and queries relationships between entities for contextual understanding
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
import logging
from datetime import datetime

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available - using fallback graph implementation")

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Manages entity relationships and contextual knowledge using graph structures
    """
    
    def __init__(self, data_file: str = "data/knowledge_graph.json"):
        self.data_file = data_file
        self._ensure_data_directory()
        
        # Initialize graph
        if NETWORKX_AVAILABLE:
            self.graph = nx.Graph()
        else:
            self.graph = FallbackGraph()
        
        # Entity categories for organization
        self.entity_categories = {
            'person': ['user', 'friend', 'family', 'colleague'],
            'location': ['city', 'country', 'place', 'address'],
            'organization': ['company', 'school', 'university', 'team'],
            'technology': ['programming_language', 'framework', 'tool', 'platform'],
            'topic': ['subject', 'interest', 'hobby', 'skill'],
            'time': ['date', 'time', 'event', 'reminder'],
            'preference': ['like', 'dislike', 'favorite', 'preference']
        }
        
        # Relationship types with weights
        self.relationship_types = {
            'personal': {
                'friends_with': 0.9,
                'family_member': 0.95,
                'colleague_of': 0.7,
                'knows': 0.5
            },
            'geographical': {
                'lives_in': 0.8,
                'works_in': 0.7,
                'visited': 0.6,
                'from': 0.8
            },
            'professional': {
                'works_at': 0.8,
                'studied_at': 0.7,
                'skilled_in': 0.6,
                'interested_in': 0.5
            },
            'preferential': {
                'likes': 0.6,
                'dislikes': 0.6,
                'prefers': 0.7,
                'uses': 0.5
            },
            'temporal': {
                'happened_on': 0.8,
                'scheduled_for': 0.9,
                'remind_about': 0.7
            }
        }
        
        # Entity extraction patterns
        self.extraction_patterns = {
            'person': [r'my (friend|brother|sister|mother|father|colleague) (\w+)', 
                      r'(\w+) (is my|works with me)'],
            'location': [r'in (\w+)', r'from (\w+)', r'to (\w+)', r'at (\w+)'],
            'organization': [r'works? at (\w+)', r'company called (\w+)', r'studied at (\w+)'],
            'preference': [r'I like (\w+)', r'I love (\w+)', r'I hate (\w+)', r'I prefer (\w+)']
        }
        
        self.entity_count = 0
        self.relationship_count = 0
        self.load_knowledge_graph()
        
        logger.info("KnowledgeGraph initialized")
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def load_knowledge_graph(self) -> None:
        """Load knowledge graph from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct graph from saved data
                entities = data.get('entities', {})
                relationships = data.get('relationships', [])
                
                for entity_id, entity_data in entities.items():
                    self._add_entity_direct(entity_id, entity_data)
                
                for rel in relationships:
                    self.add_relation(
                        rel['source'],
                        rel['relation'],
                        rel['target'],
                        rel.get('weight', 0.5),
                        rel.get('category', 'general')
                    )
                
                self.entity_count = data.get('entity_count', 0)
                self.relationship_count = data.get('relationship_count', 0)
                
                logger.debug(f"Loaded knowledge graph with {self.entity_count} entities and {self.relationship_count} relationships")
            else:
                logger.info("No existing knowledge graph found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    def save_knowledge_graph(self) -> bool:
        """Save knowledge graph to JSON file"""
        try:
            # Extract graph data
            entities = {}
            relationships = []
            
            if NETWORKX_AVAILABLE:
                for node in self.graph.nodes():
                    entities[node] = self.graph.nodes[node]
                
                for edge in self.graph.edges(data=True):
                    relationships.append({
                        'source': edge[0],
                        'target': edge[1],
                        'relation': edge[2].get('relation', 'related_to'),
                        'weight': edge[2].get('weight', 0.5),
                        'category': edge[2].get('category', 'general')
                    })
            else:
                entities = self.graph.entities
                relationships = self.graph.relationships
            
            data = {
                'entities': entities,
                'relationships': relationships,
                'entity_count': self.entity_count,
                'relationship_count': self.relationship_count,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Knowledge graph saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def _add_entity_direct(self, entity_id: str, entity_data: Dict) -> None:
        """Add entity directly to graph (used during loading)"""
        if NETWORKX_AVAILABLE:
            self.graph.add_node(entity_id, **entity_data)
        else:
            self.graph.add_node(entity_id, entity_data)
    
    def add_entity(self, entity: str, entity_type: str = 'general', properties: Dict = None) -> str:
        """
        Add an entity to the knowledge graph
        
        Args:
            entity: Entity name/identifier
            entity_type: Type of entity
            properties: Additional properties
            
        Returns:
            Entity ID
        """
        entity_id = self._normalize_entity_name(entity)
        properties = properties or {}
        
        entity_data = {
            'name': entity,
            'type': entity_type,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            **properties
        }
        
        if NETWORKX_AVAILABLE:
            if entity_id not in self.graph:
                self.graph.add_node(entity_id, **entity_data)
                self.entity_count += 1
            else:
                # Update existing entity
                current_data = self.graph.nodes[entity_id]
                current_data.update(entity_data)
        else:
            self.graph.add_node(entity_id, entity_data)
            self.entity_count += 1
        
        logger.debug(f"Added entity: {entity_id} ({entity_type})")
        return entity_id
    
    def add_relation(self, entity1: str, relation: str, entity2: str, 
                    weight: float = 0.5, category: str = 'general') -> bool:
        """
        Add a relationship between two entities
        
        Args:
            entity1: First entity
            relation: Relationship type
            entity2: Second entity
            weight: Relationship strength (0.0-1.0)
            category: Relationship category
            
        Returns:
            Success status
        """
        try:
            entity1_id = self._normalize_entity_name(entity1)
            entity2_id = self._normalize_entity_name(entity2)
            
            # Ensure entities exist
            if entity1_id not in self.graph:
                self.add_entity(entity1, 'auto_detected')
            if entity2_id not in self.graph:
                self.add_entity(entity2, 'auto_detected')
            
            edge_data = {
                'relation': relation,
                'weight': max(0.0, min(1.0, weight)),
                'category': category,
                'created_at': datetime.now().isoformat()
            }
            
            if NETWORKX_AVAILABLE:
                self.graph.add_edge(entity1_id, entity2_id, **edge_data)
            else:
                self.graph.add_edge(entity1_id, entity2_id, edge_data)
            
            self.relationship_count += 1
            logger.debug(f"Added relation: {entity1_id} -[{relation}]-> {entity2_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relation: {e}")
            return False
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistent storage"""
        return name.lower().strip().replace(' ', '_')
    
    def extract_entities_from_text(self, text: str, user_id: str = 'default') -> List[Dict]:
        """
        Extract entities and relationships from text
        
        Args:
            text: Input text to analyze
            user_id: User identifier for context
            
        Returns:
            List of extracted entities
        """
        entities_found = []
        text_lower = text.lower()
        
        for category, patterns in self.extraction_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entity_name = match.group(1) if match.groups() else match.group(0)
                    if len(entity_name) > 2:  # Avoid very short entities
                        entity_id = self.add_entity(entity_name, category)
                        entities_found.append({
                            'entity': entity_name,
                            'type': category,
                            'entity_id': entity_id,
                            'context': text
                        })
        
        # Link user to extracted entities
        user_entity_id = f"user_{user_id}"
        self.add_entity(user_id, 'person', {'user_id': user_id})
        
        for entity in entities_found:
            self.add_relation(user_entity_id, 'mentioned', entity['entity_id'], 0.3, 'contextual')
        
        return entities_found
    
    def query_graph(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Query the knowledge graph for entities and relationships
        
        Args:
            query: Search query or entity name
            max_results: Maximum number of results to return
            
        Returns:
            Query results dictionary
        """
        query_normalized = self._normalize_entity_name(query)
        results = {
            'query': query,
            'entities_found': [],
            'relationships_found': [],
            'related_entities': [],
            'suggestions': []
        }
        
        # Search for exact entity match
        if NETWORKX_AVAILABLE:
            if query_normalized in self.graph:
                entity_data = self.graph.nodes[query_normalized]
                results['entities_found'].append({
                    'entity': query_normalized,
                    'data': entity_data
                })
                
                # Get relationships for this entity
                neighbors = list(self.graph.neighbors(query_normalized))
                for neighbor in neighbors[:max_results]:
                    edge_data = self.graph[query_normalized][neighbor]
                    neighbor_data = self.graph.nodes[neighbor]
                    
                    results['relationships_found'].append({
                        'source': query_normalized,
                        'target': neighbor,
                        'relation': edge_data.get('relation', 'related_to'),
                        'weight': edge_data.get('weight', 0.5),
                        'target_data': neighbor_data
                    })
            
            # Search for partial matches
            for entity in self.graph.nodes():
                if query_normalized in entity or query.lower() in entity:
                    entity_data = self.graph.nodes[entity]
                    results['suggestions'].append({
                        'entity': entity,
                        'data': entity_data,
                        'match_type': 'partial'
                    })
        else:
            # Fallback implementation
            if query_normalized in self.graph.entities:
                entity_data = self.graph.entities[query_normalized]
                results['entities_found'].append({
                    'entity': query_normalized,
                    'data': entity_data
                })
                
                # Get relationships
                for rel in self.graph.relationships:
                    if rel['source'] == query_normalized or rel['target'] == query_normalized:
                        other_entity = rel['target'] if rel['source'] == query_normalized else rel['source']
                        results['relationships_found'].append({
                            'source': rel['source'],
                            'target': rel['target'],
                            'relation': rel['relation'],
                            'weight': rel.get('weight', 0.5),
                            'target_data': self.graph.entities.get(other_entity, {})
                        })
            
            # Partial matches
            for entity, data in self.graph.entities.items():
                if query_normalized in entity or query.lower() in data.get('name', '').lower():
                    results['suggestions'].append({
                        'entity': entity,
                        'data': data,
                        'match_type': 'partial'
                    })
        
        # Limit results
        results['suggestions'] = results['suggestions'][:max_results]
        results['relationships_found'] = results['relationships_found'][:max_results]
        
        return results
    
    def find_related_entities(self, entity: str, relation_type: str = None, 
                            min_weight: float = 0.3) -> List[Dict]:
        """
        Find entities related to a given entity
        
        Args:
            entity: Source entity
            relation_type: Specific relation type to filter by
            min_weight: Minimum relationship weight
            
        Returns:
            List of related entities
        """
        entity_id = self._normalize_entity_name(entity)
        related = []
        
        if NETWORKX_AVAILABLE:
            if entity_id not in self.graph:
                return related
            
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph[entity_id][neighbor]
                
                if (edge_data.get('weight', 0) >= min_weight and 
                    (relation_type is None or edge_data.get('relation') == relation_type)):
                    
                    neighbor_data = self.graph.nodes[neighbor]
                    related.append({
                        'entity': neighbor,
                        'relation': edge_data.get('relation', 'related_to'),
                        'weight': edge_data.get('weight', 0.5),
                        'data': neighbor_data
                    })
        else:
            for rel in self.graph.relationships:
                if (rel['source'] == entity_id or rel['target'] == entity_id) and rel.get('weight', 0) >= min_weight:
                    if relation_type is None or rel.get('relation') == relation_type:
                        other_entity = rel['target'] if rel['source'] == entity_id else rel['source']
                        related.append({
                            'entity': other_entity,
                            'relation': rel['relation'],
                            'weight': rel.get('weight', 0.5),
                            'data': self.graph.entities.get(other_entity, {})
                        })
        
        # Sort by weight (highest first)
        related.sort(key=lambda x: x['weight'], reverse=True)
        return related
    
    def get_entity_network(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get the network of entities around a central entity
        
        Args:
            entity: Central entity
            depth: How many levels to explore
            
        Returns:
            Network structure dictionary
        """
        entity_id = self._normalize_entity_name(entity)
        network = {
            'central_entity': entity_id,
            'depth': depth,
            'nodes': [],
            'edges': []
        }
        
        if NETWORKX_AVAILABLE:
            if entity_id not in self.graph:
                return network
            
            # Use BFS to explore network
            visited = set()
            queue = deque([(entity_id, 0)])
            
            while queue:
                current_entity, current_depth = queue.popleft()
                
                if current_entity in visited or current_depth > depth:
                    continue
                
                visited.add(current_entity)
                network['nodes'].append({
                    'id': current_entity,
                    'data': self.graph.nodes[current_entity],
                    'depth': current_depth
                })
                
                if current_depth < depth:
                    for neighbor in self.graph.neighbors(current_entity):
                        if neighbor not in visited:
                            queue.append((neighbor, current_depth + 1))
                            
                            edge_data = self.graph[current_entity][neighbor]
                            network['edges'].append({
                                'source': current_entity,
                                'target': neighbor,
                                'data': edge_data
                            })
        else:
            # Fallback implementation
            visited = set()
            queue = deque([(entity_id, 0)])
            
            while queue:
                current_entity, current_depth = queue.popleft()
                
                if current_entity in visited or current_depth > depth:
                    continue
                
                visited.add(current_entity)
                if current_entity in self.graph.entities:
                    network['nodes'].append({
                        'id': current_entity,
                        'data': self.graph.entities[current_entity],
                        'depth': current_depth
                    })
                
                if current_depth < depth:
                    for rel in self.graph.relationships:
                        if rel['source'] == current_entity and rel['target'] not in visited:
                            queue.append((rel['target'], current_depth + 1))
                            network['edges'].append({
                                'source': rel['source'],
                                'target': rel['target'],
                                'data': rel
                            })
                        elif rel['target'] == current_entity and rel['source'] not in visited:
                            queue.append((rel['source'], current_depth + 1))
                            network['edges'].append({
                                'source': rel['source'],
                                'target': rel['target'],
                                'data': rel
                            })
        
        return network
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if NETWORKX_AVAILABLE:
            return {
                'total_entities': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges(),
                'entity_categories': self._get_entity_categories(),
                'relationship_categories': self._get_relationship_categories(),
                'most_connected_entities': self._get_most_connected_entities(),
                'graph_density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
            }
        else:
            return {
                'total_entities': len(self.graph.entities),
                'total_relationships': len(self.graph.relationships),
                'entity_categories': self._get_entity_categories(),
                'relationship_categories': self._get_relationship_categories(),
                'most_connected_entities': self._get_most_connected_entities(),
                'graph_density': 'N/A (fallback mode)'
            }
    
    def _get_entity_categories(self) -> Dict[str, int]:
        """Count entities by category"""
        categories = defaultdict(int)
        
        if NETWORKX_AVAILABLE:
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                category = node_data.get('type', 'unknown')
                categories[category] += 1
        else:
            for entity_data in self.graph.entities.values():
                category = entity_data.get('type', 'unknown')
                categories[category] += 1
        
        return dict(categories)
    
    def _get_relationship_categories(self) -> Dict[str, int]:
        """Count relationships by category"""
        categories = defaultdict(int)
        
        if NETWORKX_AVAILABLE:
            for edge in self.graph.edges(data=True):
                category = edge[2].get('category', 'general')
                categories[category] += 1
        else:
            for rel in self.graph.relationships:
                category = rel.get('category', 'general')
                categories[category] += 1
        
        return dict(categories)
    
    def _get_most_connected_entities(self, top_n: int = 5) -> List[Dict]:
        """Get entities with most connections"""
        if NETWORKX_AVAILABLE:
            degrees = [(node, deg) for node, deg in self.graph.degree()]
            degrees.sort(key=lambda x: x[1], reverse=True)
            
            return [{'entity': node, 'connections': deg} for node, deg in degrees[:top_n]]
        else:
            # Fallback: count relationships per entity
            connection_count = defaultdict(int)
            for rel in self.graph.relationships:
                connection_count[rel['source']] += 1
                connection_count[rel['target']] += 1
            
            sorted_entities = sorted(connection_count.items(), key=lambda x: x[1], reverse=True)
            return [{'entity': entity, 'connections': count} for entity, count in sorted_entities[:top_n]]

# Fallback Graph Implementation for when NetworkX is not available
class FallbackGraph:
    """Simple graph implementation when NetworkX is not available"""
    
    def __init__(self):
        self.entities = {}
        self.relationships = []
    
    def add_node(self, node_id: str, data: Dict) -> None:
        """Add a node to the graph"""
        self.entities[node_id] = data
    
    def add_edge(self, source: str, target: str, data: Dict) -> None:
        """Add an edge to the graph"""
        self.relationships.append({
            'source': source,
            'target': target,
            **data
        })
    
    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph"""
        return node_id in self.entities

# Utility function for easy integration
def get_knowledge_graph() -> KnowledgeGraph:
    """Get initialized knowledge graph instance"""
    return KnowledgeGraph()

# Test function
def test_knowledge_graph():
    """Test the knowledge graph functionality"""
    kg = KnowledgeGraph("test_knowledge_graph.json")
    
    # Test adding entities and relationships
    print("Knowledge Graph Test:")
    print("=" * 50)
    
    # Add some entities
    kg.add_entity("Alice", "person", {"age": 30, "profession": "engineer"})
    kg.add_entity("Bob", "person", {"age": 25, "profession": "designer"})
    kg.add_entity("Python", "technology", {"category": "programming_language"})
    kg.add_entity("Delhi", "location", {"country": "India"})
    
    # Add relationships
    kg.add_relation("Alice", "friends_with", "Bob", 0.8, "personal")
    kg.add_relation("Alice", "skilled_in", "Python", 0.9, "professional")
    kg.add_relation("Alice", "lives_in", "Delhi", 0.7, "geographical")
    kg.add_relation("Bob", "interested_in", "Python", 0.6, "professional")
    
    # Test queries
    print("Query for 'Alice':")
    result = kg.query_graph("Alice")
    print(f"Entities found: {len(result['entities_found'])}")
    print(f"Relationships: {len(result['relationships_found'])}")
    
    for rel in result['relationships_found']:
        print(f"  {rel['source']} -[{rel['relation']}]-> {rel['target']}")
    
    # Test entity extraction
    print("\nEntity extraction from text:")
    text = "My friend Charlie works at Google and loves Python programming"
    entities = kg.extract_entities_from_text(text, "test_user")
    print(f"Extracted entities: {[e['entity'] for e in entities]}")
    
    # Test network exploration
    print("\nNetwork around 'Alice':")
    network = kg.get_entity_network("Alice", depth=2)
    print(f"Network nodes: {len(network['nodes'])}")
    print(f"Network edges: {len(network['edges'])}")
    
    # Test statistics
    stats = kg.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Entity categories: {stats['entity_categories']}")
    print(f"Most connected: {stats['most_connected_entities']}")
    
    # Cleanup test file
    if os.path.exists("test_knowledge_graph.json"):
        os.remove("test_knowledge_graph.json")

if __name__ == "__main__":
    test_knowledge_graph()