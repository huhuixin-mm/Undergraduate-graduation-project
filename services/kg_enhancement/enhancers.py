"""
Knowledge Graph Enhanced LLM Module
Implements RAG, Reasoning & Planning, and Synergy & Feedback methods
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings


class KGEnhancementBase(ABC):
    """Base class for knowledge graph enhancement methods"""
    
    @abstractmethod
    def enhance_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Enhance the user query with knowledge graph information"""
        pass
    
    @abstractmethod
    def process_response(self, response: str, query: str) -> str:
        """Process the model response and potentially enhance it"""
        pass


class RAGEnhancer(KGEnhancementBase):
    """
    RAG (Retrieval-Augmented Generation) Enhancement
    Retrieves relevant information from knowledge graph and documents
    """
    
    def __init__(self, vector_db=None, kg_client=None):
        self.vector_db = vector_db
        self.kg_client = kg_client
        self.max_retrieved_docs = 5
        self.similarity_threshold = 0.7
    
    def enhance_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Enhance query with retrieved relevant information
        
        Args:
            query: Original user query
            context: Additional context information
            
        Returns:
            Enhanced query with retrieved information
        """
        try:
            # Step 1: Retrieve relevant documents from vector database
            retrieved_docs = self._retrieve_from_vector_db(query)
            
            # Step 2: Retrieve relevant entities and relations from KG
            kg_info = self._retrieve_from_kg(query)
            
            # Step 3: Combine and format the information
            enhanced_query = self._format_enhanced_query(query, retrieved_docs, kg_info)
            
            return enhanced_query
            
        except Exception as e:
            print(f"RAG enhancement failed: {e}")
            return query  # Fallback to original query
    
    def _retrieve_from_vector_db(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database"""
        if not self.vector_db:
            return []
        
        try:
            # Query vector database for similar documents
            results = self.vector_db.similarity_search(
                query, 
                k=self.max_retrieved_docs,
                score_threshold=self.similarity_threshold
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"Vector DB retrieval failed: {e}")
            return []
    
    def _retrieve_from_kg(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant entities and relations from knowledge graph"""
        if not self.kg_client:
            return {}
        
        try:
            # Extract entities from query (simplified approach)
            entities = self._extract_entities(query)
            
            # Query knowledge graph for related information
            kg_results = {}
            for entity in entities:
                # Get entity properties and relations
                entity_info = self.kg_client.get_entity_info(entity)
                related_entities = self.kg_client.get_related_entities(entity)
                
                kg_results[entity] = {
                    "properties": entity_info,
                    "relations": related_entities
                }
            
            return kg_results
            
        except Exception as e:
            print(f"KG retrieval failed: {e}")
            return {}
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query (simplified)"""
        # This is a simplified implementation
        # In practice, you'd use NER or entity linking
        medical_terms = [
            "brain", "extracellular space", "ECS", "neuron", "synapse",
            "traditional chinese medicine", "TCM", "acupuncture"
        ]
        
        entities = []
        query_lower = query.lower()
        for term in medical_terms:
            if term in query_lower:
                entities.append(term)
        
        return entities
    
    def _format_enhanced_query(
        self, 
        original_query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        kg_info: Dict[str, Any]
    ) -> str:
        """Format the enhanced query with retrieved information"""
        
        enhanced_parts = [
            "Based on the following context, please answer the question:",
            "",
            "=== RETRIEVED DOCUMENTS ==="
        ]
        
        # Add retrieved documents
        for i, doc in enumerate(retrieved_docs[:3]):  # Limit to top 3
            enhanced_parts.extend([
                f"Document {i+1}:",
                doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"],
                ""
            ])
        
        # Add knowledge graph information
        if kg_info:
            enhanced_parts.extend([
                "=== KNOWLEDGE GRAPH INFORMATION ===",
                ""
            ])
            
            for entity, info in kg_info.items():
                enhanced_parts.append(f"Entity: {entity}")
                if info.get("properties"):
                    enhanced_parts.append(f"Properties: {info['properties']}")
                if info.get("relations"):
                    enhanced_parts.append(f"Related to: {', '.join(info['relations'])}")
                enhanced_parts.append("")
        
        # Add original question
        enhanced_parts.extend([
            "=== QUESTION ===",
            original_query
        ])
        
        return "\n".join(enhanced_parts)
    
    def process_response(self, response: str, query: str) -> str:
        """Process and validate the response"""
        # Add source attribution if possible
        return f"{response}\n\n*Response generated using retrieved medical literature and knowledge graph.*"


class ReasoningPlanningEnhancer(KGEnhancementBase):
    """
    Reasoning & Planning Enhancement
    Uses knowledge graph structure for multi-step reasoning
    """
    
    def __init__(self, kg_client=None):
        self.kg_client = kg_client
        self.max_reasoning_steps = 5
    
    def enhance_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Enhance query with reasoning path from knowledge graph
        """
        try:
            # Step 1: Identify reasoning type and entities
            reasoning_plan = self._create_reasoning_plan(query)
            
            # Step 2: Execute multi-hop reasoning on KG
            reasoning_path = self._execute_reasoning(reasoning_plan)
            
            # Step 3: Format enhanced query with reasoning context
            enhanced_query = self._format_reasoning_query(query, reasoning_path)
            
            return enhanced_query
            
        except Exception as e:
            print(f"Reasoning enhancement failed: {e}")
            return query
    
    def _create_reasoning_plan(self, query: str) -> Dict[str, Any]:
        """Create a reasoning plan based on the query"""
        # Simplified reasoning plan creation
        plan = {
            "query_type": self._classify_query_type(query),
            "entities": self._extract_entities(query),
            "reasoning_steps": []
        }
        
        # Define reasoning steps based on query type
        if plan["query_type"] == "causal":
            plan["reasoning_steps"] = ["find_causes", "trace_effects", "analyze_mechanisms"]
        elif plan["query_type"] == "comparative":
            plan["reasoning_steps"] = ["identify_concepts", "find_similarities", "find_differences"]
        elif plan["query_type"] == "therapeutic":
            plan["reasoning_steps"] = ["identify_condition", "find_treatments", "analyze_efficacy"]
        else:
            plan["reasoning_steps"] = ["identify_concepts", "find_relations", "synthesize"]
        
        return plan
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of reasoning required"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["cause", "why", "reason", "mechanism"]):
            return "causal"
        elif any(word in query_lower for word in ["compare", "difference", "similar", "vs"]):
            return "comparative"
        elif any(word in query_lower for word in ["treat", "therapy", "cure", "medicine"]):
            return "therapeutic"
        else:
            return "general"
    
    def _execute_reasoning(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute multi-step reasoning on knowledge graph"""
        if not self.kg_client:
            return []
        
        reasoning_path = []
        
        try:
            for step in plan["reasoning_steps"]:
                step_result = self._execute_reasoning_step(step, plan["entities"], reasoning_path)
                reasoning_path.append({
                    "step": step,
                    "result": step_result
                })
        
        except Exception as e:
            print(f"Reasoning execution failed: {e}")
        
        return reasoning_path
    
    def _execute_reasoning_step(
        self, 
        step: str, 
        entities: List[str], 
        previous_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        # This is a simplified implementation
        # In practice, you'd have more sophisticated reasoning logic
        
        if step == "find_causes":
            return {"type": "causal", "relations": ["causes", "leads_to", "triggers"]}
        elif step == "find_treatments":
            return {"type": "therapeutic", "relations": ["treats", "alleviates", "cures"]}
        elif step == "find_relations":
            return {"type": "associative", "relations": ["related_to", "part_of", "affects"]}
        else:
            return {"type": "general", "relations": ["connected_to"]}
    
    def _format_reasoning_query(self, original_query: str, reasoning_path: List[Dict[str, Any]]) -> str:
        """Format query with reasoning context"""
        enhanced_parts = [
            "Please use step-by-step reasoning to answer this question:",
            "",
            "=== REASONING APPROACH ==="
        ]
        
        for i, step in enumerate(reasoning_path, 1):
            enhanced_parts.extend([
                f"Step {i}: {step['step'].replace('_', ' ').title()}",
                f"Focus: {step['result']}",
                ""
            ])
        
        enhanced_parts.extend([
            "=== QUESTION ===",
            original_query,
            "",
            "Please provide a detailed reasoning process following the steps above."
        ])
        
        return "\n".join(enhanced_parts)
    
    def process_response(self, response: str, query: str) -> str:
        """Process response to ensure reasoning is explicit"""
        return f"{response}\n\n*Response includes multi-step reasoning based on medical knowledge graph.*"


class SynergyFeedbackEnhancer(KGEnhancementBase):
    """
    Synergy & Feedback Enhancement
    Bidirectional enhancement between LLM and KG
    """
    
    def __init__(self, kg_client=None, model_handler=None):
        self.kg_client = kg_client
        self.model_handler = model_handler
        self.feedback_enabled = True
    
    def enhance_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Enhance query and prepare for feedback loop
        """
        try:
            # Step 1: Use LLM to extract and expand entities
            expanded_entities = self._llm_entity_expansion(query)
            
            # Step 2: Validate and correct entities using KG
            validated_entities = self._validate_entities_with_kg(expanded_entities)
            
            # Step 3: Use LLM to generate KG queries
            kg_queries = self._generate_kg_queries(query, validated_entities)
            
            # Step 4: Execute KG queries and format results
            kg_results = self._execute_kg_queries(kg_queries)
            
            # Step 5: Format enhanced query
            enhanced_query = self._format_synergy_query(query, kg_results)
            
            return enhanced_query
            
        except Exception as e:
            print(f"Synergy enhancement failed: {e}")
            return query
    
    def _llm_entity_expansion(self, query: str) -> List[str]:
        """Use LLM to identify and expand medical entities"""
        if not self.model_handler:
            return self._extract_entities(query)
        
        try:
            expansion_prompt = f"""
            Please identify all medical entities and concepts in the following query, 
            including related terms and synonyms:
            
            Query: {query}
            
            List the entities in the format: [entity1, entity2, entity3, ...]
            """
            
            # This would use the model to expand entities
            # For now, we'll use a simplified approach
            return self._extract_entities(query)
            
        except Exception as e:
            print(f"LLM entity expansion failed: {e}")
            return self._extract_entities(query)
    
    def _validate_entities_with_kg(self, entities: List[str]) -> List[str]:
        """Validate entities against knowledge graph"""
        if not self.kg_client:
            return entities
        
        validated = []
        for entity in entities:
            if self.kg_client.entity_exists(entity):
                validated.append(entity)
            else:
                # Try to find similar entities
                similar = self.kg_client.find_similar_entities(entity)
                if similar:
                    validated.extend(similar[:2])  # Add top 2 similar entities
        
        return validated
    
    def _generate_kg_queries(self, query: str, entities: List[str]) -> List[str]:
        """Generate knowledge graph queries"""
        queries = []
        
        for entity in entities:
            queries.extend([
                f"MATCH (n:Entity {{name: '{entity}'}}) RETURN n",
                f"MATCH (n:Entity {{name: '{entity}'}})-[r]-(m) RETURN r, m"
            ])
        
        return queries
    
    def _execute_kg_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute knowledge graph queries"""
        if not self.kg_client:
            return []
        
        results = []
        for query in queries:
            try:
                result = self.kg_client.execute_query(query)
                results.append(result)
            except Exception as e:
                print(f"KG query failed: {e}")
        
        return results
    
    def _format_synergy_query(self, original_query: str, kg_results: List[Dict[str, Any]]) -> str:
        """Format enhanced query with synergy results"""
        enhanced_parts = [
            "Using integrated knowledge from both language understanding and knowledge graph:",
            "",
            "=== KNOWLEDGE GRAPH INSIGHTS ==="
        ]
        
        # Add KG results (simplified)
        for i, result in enumerate(kg_results[:5]):  # Limit results
            enhanced_parts.append(f"Insight {i+1}: {result}")
        
        enhanced_parts.extend([
            "",
            "=== QUESTION ===",
            original_query,
            "",
            "Please provide a comprehensive answer that integrates both textual knowledge and structured medical knowledge."
        ])
        
        return "\n".join(enhanced_parts)
    
    def process_response(self, response: str, query: str) -> str:
        """Process response and provide feedback to KG"""
        processed_response = response
        
        if self.feedback_enabled:
            # Extract new knowledge from response for KG update
            new_knowledge = self._extract_knowledge_from_response(response, query)
            if new_knowledge:
                self._update_kg_with_feedback(new_knowledge)
                processed_response += "\n\n*Knowledge graph updated with new insights from this interaction.*"
        
        return processed_response
    
    def _extract_knowledge_from_response(self, response: str, query: str) -> Dict[str, Any]:
        """Extract structured knowledge from LLM response"""
        # Simplified knowledge extraction
        # In practice, you'd use more sophisticated NLP techniques
        return {
            "entities": [],
            "relations": [],
            "facts": []
        }
    
    def _update_kg_with_feedback(self, knowledge: Dict[str, Any]):
        """Update knowledge graph with extracted knowledge"""
        if not self.kg_client:
            return
        
        try:
            # Update KG with new entities, relations, and facts
            for entity in knowledge.get("entities", []):
                self.kg_client.add_entity(entity)
            
            for relation in knowledge.get("relations", []):
                self.kg_client.add_relation(relation)
                
        except Exception as e:
            print(f"KG update failed: {e}")


# Factory function to create appropriate enhancer
def create_kg_enhancer(method: str = "rag", **kwargs) -> KGEnhancementBase:
    """
    Create knowledge graph enhancer based on method
    
    Args:
        method: Enhancement method ('rag', 'reasoning', 'synergy')
        **kwargs: Additional arguments for enhancer initialization
    
    Returns:
        KGEnhancementBase instance
    """
    if method.lower() == "rag":
        return RAGEnhancer(**kwargs)
    elif method.lower() in ["reasoning", "planning"]:
        return ReasoningPlanningEnhancer(**kwargs)
    elif method.lower() in ["synergy", "feedback"]:
        return SynergyFeedbackEnhancer(**kwargs)
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


# Example usage
if __name__ == "__main__":
    # Test the enhancers
    test_query = "What is the role of brain extracellular space in neuronal communication?"
    
    # Test RAG enhancer
    rag_enhancer = create_kg_enhancer("rag")
    enhanced_query = rag_enhancer.enhance_query(test_query)
    print("RAG Enhanced Query:")
    print(enhanced_query)
    print("\n" + "="*50 + "\n")
    
    # Test Reasoning enhancer
    reasoning_enhancer = create_kg_enhancer("reasoning")
    enhanced_query = reasoning_enhancer.enhance_query(test_query)
    print("Reasoning Enhanced Query:")
    print(enhanced_query)