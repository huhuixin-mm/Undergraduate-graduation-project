"""
Knowledge Graph Client for Neo4j
Handles connections and queries to the knowledge graph
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings


class KnowledgeGraphClient:
    """Client for interacting with Neo4j knowledge graph"""
    
    def __init__(self, 
                 uri: str = None,
                 user: str = None,
                 password: str = None):
        
        self.uri = uri or settings.kg.neo4j_uri
        self.user = user or settings.kg.neo4j_user
        self.password = password or settings.kg.neo4j_password
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message")
                message = result.single()["message"]
                print(f"✅ Knowledge graph connected: {message}")
                
        except Exception as e:
            print(f"❌ Failed to connect to knowledge graph: {e}")
            print("💡 Make sure Neo4j is running and credentials are correct")
            self.driver = None
    
    def close(self):
        """Close the connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Knowledge graph connection closed")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
        
        Returns:
            List of result records
        """
        if not self.driver:
            print("❌ No connection to knowledge graph")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return []
    
    def entity_exists(self, entity_name: str) -> bool:
        """Check if an entity exists in the knowledge graph"""
        query = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($entity_name) OR 
              toLower(n.label) = toLower($entity_name)
        RETURN count(n) > 0 as exists
        """
        
        result = self.execute_query(query, {"entity_name": entity_name})
        return result[0]["exists"] if result else False
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Get detailed information about an entity"""
        query = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($entity_name) OR 
              toLower(n.label) = toLower($entity_name)
        RETURN n
        LIMIT 1
        """
        
        result = self.execute_query(query, {"entity_name": entity_name})
        
        if result:
            return dict(result[0]["n"])
        else:
            return {}
    
    def get_related_entities(self, 
                           entity_name: str, 
                           max_depth: int = 2,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get entities related to the given entity"""
        query = f"""
        MATCH (start)
        WHERE toLower(start.name) = toLower($entity_name) OR 
              toLower(start.label) = toLower($entity_name)
        MATCH (start)-[r*1..{max_depth}]-(related)
        WHERE start <> related
        RETURN DISTINCT related.name as name, 
               related.type as type,
               type(r[0]) as relationship
        LIMIT {limit}
        """
        
        return self.execute_query(query, {"entity_name": entity_name})
    
    def find_similar_entities(self, entity_name: str, limit: int = 5) -> List[str]:
        """Find entities with similar names"""
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($entity_name) OR 
              toLower(n.label) CONTAINS toLower($entity_name)
        RETURN DISTINCT n.name as name
        LIMIT $limit
        """
        
        result = self.execute_query(query, {
            "entity_name": entity_name,
            "limit": limit
        })
        
        return [record["name"] for record in result if record["name"]]
    
    def get_path_between_entities(self, 
                                entity1: str, 
                                entity2: str,
                                max_length: int = 5) -> List[Dict[str, Any]]:
        """Find the shortest path between two entities"""
        query = f"""
        MATCH (start), (end)
        WHERE (toLower(start.name) = toLower($entity1) OR 
               toLower(start.label) = toLower($entity1)) AND
              (toLower(end.name) = toLower($entity2) OR 
               toLower(end.label) = toLower($entity2))
        MATCH path = shortestPath((start)-[*1..{max_length}]-(end))
        RETURN path
        LIMIT 1
        """
        
        result = self.execute_query(query, {
            "entity1": entity1,
            "entity2": entity2
        })
        
        return result
    
    def add_entity(self, entity_data: Dict[str, Any]) -> bool:
        """Add a new entity to the knowledge graph"""
        try:
            # Basic entity creation
            query = """
            CREATE (n:Entity)
            SET n += $properties
            RETURN id(n) as entity_id
            """
            
            result = self.execute_query(query, {"properties": entity_data})
            
            if result:
                print(f"✅ Added entity: {entity_data.get('name', 'unnamed')}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to add entity: {e}")
            return False
    
    def add_relation(self, relation_data: Dict[str, Any]) -> bool:
        """Add a new relation between entities"""
        try:
            # Find source and target entities
            source = relation_data.get("source")
            target = relation_data.get("target")
            rel_type = relation_data.get("type", "RELATED_TO")
            properties = relation_data.get("properties", {})
            
            query = f"""
            MATCH (source), (target)
            WHERE (toLower(source.name) = toLower($source) OR 
                   toLower(source.label) = toLower($source)) AND
                  (toLower(target.name) = toLower($target) OR 
                   toLower(target.label) = toLower($target))
            CREATE (source)-[r:{rel_type}]->(target)
            SET r += $properties
            RETURN id(r) as relation_id
            """
            
            result = self.execute_query(query, {
                "source": source,
                "target": target,
                "properties": properties
            })
            
            if result:
                print(f"✅ Added relation: {source} -{rel_type}-> {target}")
                return True
            else:
                print(f"❌ Could not find entities: {source}, {target}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to add relation: {e}")
            return False
    
    def search_by_properties(self, 
                           properties: Dict[str, Any],
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by properties"""
        # Build WHERE clause dynamically
        where_conditions = []
        params = {}
        
        for key, value in properties.items():
            param_name = f"prop_{key}"
            where_conditions.append(f"toLower(n.{key}) CONTAINS toLower(${param_name})")
            params[param_name] = str(value)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "true"
        
        query = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN n
        LIMIT {limit}
        """
        
        result = self.execute_query(query, params)
        return [dict(record["n"]) for record in result]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_types": "MATCH (n) RETURN DISTINCT labels(n) as types",
            "relationship_types": "MATCH ()-[r]->() RETURN DISTINCT type(r) as types"
        }
        
        stats = {}
        
        for stat_name, query in queries.items():
            try:
                result = self.execute_query(query)
                if stat_name in ["total_nodes", "total_relationships"]:
                    stats[stat_name] = result[0]["count"] if result else 0
                else:
                    stats[stat_name] = [record["types"] for record in result]
            except Exception as e:
                print(f"❌ Failed to get {stat_name}: {e}")
                stats[stat_name] = None
        
        return stats


# Utility functions for common medical KG operations
class MedicalKGOperations:
    """Specialized operations for medical knowledge graphs"""
    
    def __init__(self, kg_client: KnowledgeGraphClient):
        self.kg = kg_client
    
    def find_treatments_for_condition(self, condition: str) -> List[Dict[str, Any]]:
        """Find treatments for a medical condition"""
        query = """
        MATCH (condition)-[:TREATED_BY|:ALLEVIATES|:CURES]-(treatment)
        WHERE toLower(condition.name) CONTAINS toLower($condition) OR
              toLower(condition.label) CONTAINS toLower($condition)
        RETURN DISTINCT treatment.name as treatment, 
               treatment.type as type,
               treatment.description as description
        LIMIT 20
        """
        
        return self.kg.execute_query(query, {"condition": condition})
    
    def find_causes_of_condition(self, condition: str) -> List[Dict[str, Any]]:
        """Find causes of a medical condition"""
        query = """
        MATCH (cause)-[:CAUSES|:LEADS_TO|:TRIGGERS]->(condition)
        WHERE toLower(condition.name) CONTAINS toLower($condition) OR
              toLower(condition.label) CONTAINS toLower($condition)
        RETURN DISTINCT cause.name as cause,
               cause.type as type,
               cause.description as description
        LIMIT 20
        """
        
        return self.kg.execute_query(query, {"condition": condition})
    
    def find_tcm_western_connections(self, concept: str) -> List[Dict[str, Any]]:
        """Find connections between TCM and Western medicine concepts"""
        query = """
        MATCH (tcm)-[:CORRESPONDS_TO|:SIMILAR_TO|:TREATS_SAME]-(western)
        WHERE (toLower(tcm.name) CONTAINS toLower($concept) OR
               toLower(western.name) CONTAINS toLower($concept)) AND
              (tcm.category = 'TCM' OR western.category = 'Western')
        RETURN DISTINCT tcm.name as tcm_concept,
               western.name as western_concept,
               type(r) as relationship_type
        LIMIT 15
        """
        
        return self.kg.execute_query(query, {"concept": concept})


# Factory function
def create_kg_client() -> Optional[KnowledgeGraphClient]:
    """Create and return a KnowledgeGraphClient instance"""
    try:
        return KnowledgeGraphClient()
    except Exception as e:
        print(f"❌ Failed to create KG client: {e}")
        return None


if __name__ == "__main__":
    # Test the knowledge graph client
    kg_client = create_kg_client()
    
    if kg_client:
        print("\n📊 Testing knowledge graph operations:")
        
        # Test basic operations
        stats = kg_client.get_statistics()
        print(f"KG Statistics: {stats}")
        
        # Test entity search
        results = kg_client.find_similar_entities("brain")
        print(f"Similar entities to 'brain': {results}")
        
        # Test medical operations
        medical_ops = MedicalKGOperations(kg_client)
        treatments = medical_ops.find_treatments_for_condition("headache")
        print(f"Treatments for headache: {treatments}")
        
        kg_client.close()
    else:
        print("❌ Could not connect to knowledge graph")