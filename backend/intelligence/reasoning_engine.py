# Chain-of-thought processing
"""
Mickey AI - Reasoning Engine
Advanced chain-of-thought reasoning with context awareness and logical processing
"""

import logging
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import random

class ReasoningState(Enum):
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

class ReasoningStep:
    def __init__(self, step_type: str, description: str, data: Any = None):
        self.step_type = step_type
        self.description = description
        self.data = data
        self.timestamp = time.time()
        self.duration = 0.0

class ReasoningEngine:
    def __init__(self, groq_client=None, memory_manager=None):
        self.logger = logging.getLogger(__name__)
        
        # Dependencies
        self.groq_client = groq_client
        self.memory_manager = memory_manager
        
        # Reasoning state
        self.current_state = ReasoningState.COMPLETED
        self.reasoning_chain = []
        self.context_window = 10  # Number of previous interactions to consider
        
        # Cognitive parameters
        self.creativity_level = 0.7  # 0-1 scale
        self.analytical_depth = 0.8   # 0-1 scale
        self.context_sensitivity = 0.9 # 0-1 scale
        
        # Knowledge domains
        self.supported_domains = [
            "general_knowledge", "technology", "entertainment", 
            "science", "history", "personal_assistance", "humor"
        ]
        
        # Reasoning patterns
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        # Chain-of-thought templates
        self.cot_templates = self._initialize_cot_templates()
        
        # Mickey's reasoning personalities
        self.reasoning_personalities = {
            "analytical": "Mickey's analyzing this carefully... 🤔",
            "creative": "Mickey's thinking outside the box! 🎨",
            "practical": "Mickey's finding the most practical solution! 🔧",
            "humorous": "Mickey's adding some fun to this! 🎪"
        }
        
        self.logger.info("🧠 Reasoning Engine initialized - Ready for smart thinking!")

    def _initialize_reasoning_patterns(self) -> Dict[str, Any]:
        """Initialize different reasoning patterns"""
        return {
            "deductive": {
                "description": "Logical deduction from general principles",
                "steps": ["identify_premises", "apply_rules", "draw_conclusion"],
                "confidence_threshold": 0.8
            },
            "inductive": {
                "description": "Generalizing from specific observations",
                "steps": ["gather_evidence", "identify_patterns", "form_generalization"],
                "confidence_threshold": 0.6
            },
            "abductive": {
                "description": "Inference to the best explanation",
                "steps": ["observe_facts", "generate_hypotheses", "select_best_explanation"],
                "confidence_threshold": 0.7
            },
            "analogical": {
                "description": "Reasoning by analogy and comparison",
                "steps": ["identify_analogy", "map_correspondences", "apply_insights"],
                "confidence_threshold": 0.65
            },
            "practical": {
                "description": "Pragmatic problem-solving",
                "steps": ["define_problem", "evaluate_options", "choose_solution"],
                "confidence_threshold": 0.75
            }
        }

    def _initialize_cot_templates(self) -> Dict[str, str]:
        """Initialize chain-of-thought reasoning templates"""
        return {
            "problem_solving": """
            Let me think through this step by step:
            
            1. First, I need to understand the problem: {problem}
            2. The key elements are: {key_elements}
            3. I know that: {existing_knowledge}
            4. Based on this, I can reason: {reasoning_steps}
            5. Therefore, the solution is: {conclusion}
            """,
            
            "decision_making": """
            I need to make a careful decision:
            
            Option Analysis:
            {options_analysis}
            
            Pros and Cons:
            {pros_cons}
            
            Best Choice Reasoning:
            {choice_reasoning}
            
            Final Decision: {decision}
            """,
            
            "explanation": """
            Let me explain this clearly:
            
            The main concept is: {main_concept}
            How it works: {mechanism}
            Why it matters: {significance}
            Example: {example}
            
            In summary: {summary}
            """,
            
            "creative_thinking": """
            Time for some creative thinking!
            
            Original idea: {original_idea}
            Alternative perspectives: {perspectives}
            Making connections: {connections}
            Enhanced idea: {enhanced_idea}
            
            Creative insight: {insight}
            """
        }

    async def process_message(self, message: str, context: List[Dict] = None, 
                            user_id: str = "default") -> Dict[str, Any]:
        """
        Process a message with advanced reasoning
        
        Args:
            message: User message to process
            context: Conversation context
            user_id: User identifier for personalization
            
        Returns:
            Dictionary with reasoning result
        """
        start_time = time.time()
        self.current_state = ReasoningState.ANALYZING
        self.reasoning_chain = []
        
        try:
            # Step 1: Analyze the message
            analysis = await self._analyze_message(message, context, user_id)
            self._add_reasoning_step("analysis", "Analyzed message intent and context", analysis)
            
            # Step 2: Determine reasoning approach
            reasoning_approach = self._select_reasoning_approach(analysis)
            self._add_reasoning_step("approach_selection", f"Selected {reasoning_approach} reasoning approach")
            
            # Step 3: Apply reasoning
            reasoning_result = await self._apply_reasoning(message, analysis, reasoning_approach, context)
            self._add_reasoning_step("reasoning", f"Applied {reasoning_approach} reasoning", reasoning_result)
            
            # Step 4: Generate response
            response = await self._generate_response(reasoning_result, analysis, user_id)
            self._add_reasoning_step("response_generation", "Generated final response")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.current_state = ReasoningState.COMPLETED
            
            # Compile reasoning report
            reasoning_report = self._compile_reasoning_report(processing_time)
            
            self.logger.info(f"Reasoning completed in {processing_time:.2f}s - Approach: {reasoning_approach}")
            
            return {
                'success': True,
                'response': response['text'],
                'reasoning_approach': reasoning_approach,
                'confidence': reasoning_result.get('confidence', 0.7),
                'processing_time': processing_time,
                'reasoning_chain_length': len(self.reasoning_chain),
                'reasoning_report': reasoning_report,
                'mickey_response': self._get_reasoning_personality(response.get('tone', 'analytical'))
            }
            
        except Exception as e:
            self.current_state = ReasoningState.ERROR
            self.logger.error(f"Reasoning process failed: {str(e)}")
            return self._create_error_response(f"Reasoning failed: {str(e)}")

    async def _analyze_message(self, message: str, context: List[Dict], user_id: str) -> Dict[str, Any]:
        """Analyze message for intent, sentiment, and context"""
        analysis = {
            'intent': 'unknown',
            'sentiment': 'neutral',
            'complexity': 'medium',
            'domain': 'general',
            'requires_reasoning': False,
            'key_entities': [],
            'context_references': []
        }
        
        # Basic intent detection
        intent_analysis = self._detect_intent(message)
        analysis.update(intent_analysis)
        
        # Sentiment analysis
        analysis['sentiment'] = self._analyze_sentiment(message)
        
        # Complexity assessment
        analysis['complexity'] = self._assess_complexity(message)
        
        # Domain classification
        analysis['domain'] = self._classify_domain(message)
        
        # Entity extraction
        analysis['key_entities'] = self._extract_entities(message)
        
        # Context analysis
        if context:
            analysis['context_references'] = self._analyze_context_references(message, context)
        
        # Determine if advanced reasoning is needed
        analysis['requires_reasoning'] = self._requires_advanced_reasoning(analysis)
        
        return analysis

    def _detect_intent(self, message: str) -> Dict[str, Any]:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        intents = {
            'question': any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']),
            'command': any(word in message_lower for word in ['play', 'stop', 'open', 'close', 'search', 'find']),
            'explanation': any(word in message_lower for word in ['explain', 'describe', 'tell me about']),
            'comparison': any(word in message_lower for word in ['compare', 'difference between', 'vs', 'versus']),
            'opinion': any(word in message_lower for word in ['think', 'opinion', 'view', 'believe']),
            'creative': any(word in message_lower for word in ['imagine', 'create', 'story', 'poem']),
            'humor': any(word in message_lower for word in ['joke', 'funny', 'laugh', 'humor'])
        }
        
        # Find primary intent
        primary_intent = 'conversation'  # Default
        for intent, detected in intents.items():
            if detected:
                primary_intent = intent
                break
        
        return {
            'intent': primary_intent,
            'sub_intents': [intent for intent, detected in intents.items() if detected]
        }

    def _analyze_sentiment(self, message: str) -> str:
        """Analyze message sentiment"""
        positive_words = ['good', 'great', 'awesome', 'excellent', 'amazing', 'happy', 'love', 'thanks']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'disappointed']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def _assess_complexity(self, message: str) -> str:
        """Assess message complexity"""
        word_count = len(message.split())
        sentence_count = len(re.split(r'[.!?]+', message))
        
        if word_count < 5:
            return 'simple'
        elif word_count > 20 or sentence_count > 3:
            return 'complex'
        else:
            return 'medium'

    def _classify_domain(self, message: str) -> str:
        """Classify message into knowledge domain"""
        domain_keywords = {
            'technology': ['computer', 'software', 'code', 'program', 'tech', 'ai', 'machine learning'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'research', 'experiment'],
            'entertainment': ['movie', 'music', 'game', 'entertainment', 'celebrity', 'film'],
            'history': ['history', 'historical', 'past', 'ancient', 'century'],
            'personal_assistance': ['remind', 'schedule', 'help me', 'assist', 'todo']
        }
        
        message_lower = message.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return domain
        
        return 'general_knowledge'

    def _extract_entities(self, message: str) -> List[str]:
        """Extract key entities from message"""
        # Simple entity extraction - in production, use NER
        entities = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', message)
        entities.extend(quoted)
        
        # Extract capitalized phrases (potential proper nouns)
        words = message.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entities.append(word)
        
        return list(set(entities))

    def _analyze_context_references(self, message: str, context: List[Dict]) -> List[str]:
        """Analyze references to previous context"""
        references = []
        message_lower = message.lower()
        
        for i, ctx in enumerate(context[-self.context_window:]):
            ctx_text = ctx.get('message', '').lower()
            ctx_entities = self._extract_entities(ctx_text)
            
            # Check if current message references previous context
            for entity in ctx_entities:
                if entity.lower() in message_lower:
                    references.append(f"Reference to previous message {i+1}: {entity}")
        
        return references

    def _requires_advanced_reasoning(self, analysis: Dict[str, Any]) -> bool:
        """Determine if advanced reasoning is required"""
        complex_intents = ['explanation', 'comparison', 'creative', 'opinion']
        complex_domains = ['science', 'technology', 'history']
        
        return (analysis['intent'] in complex_intents or 
                analysis['domain'] in complex_domains or
                analysis['complexity'] == 'complex')

    def _select_reasoning_approach(self, analysis: Dict[str, Any]) -> str:
        """Select appropriate reasoning approach based on analysis"""
        intent = analysis['intent']
        domain = analysis['domain']
        complexity = analysis['complexity']
        
        if intent == 'comparison':
            return 'analogical'
        elif intent == 'explanation' and domain in ['science', 'technology']:
            return 'deductive'
        elif intent == 'opinion' or complexity == 'complex':
            return 'abductive'
        elif intent == 'creative':
            return 'analogical'
        else:
            return 'practical'

    async def _apply_reasoning(self, message: str, analysis: Dict[str, Any], 
                             approach: str, context: List[Dict]) -> Dict[str, Any]:
        """Apply specific reasoning approach"""
        reasoning_methods = {
            'deductive': self._deductive_reasoning,
            'inductive': self._inductive_reasoning,
            'abductive': self._abductive_reasoning,
            'analogical': self._analogical_reasoning,
            'practical': self._practical_reasoning
        }
        
        method = reasoning_methods.get(approach, self._practical_reasoning)
        return await method(message, analysis, context)

    async def _deductive_reasoning(self, message: str, analysis: Dict[str, Any], context: List[Dict]) -> Dict[str, Any]:
        """Apply deductive reasoning (general to specific)"""
        # Use LLM for complex deductive reasoning
        if self.groq_client:
            prompt = f"""
            Apply deductive reasoning to this message: "{message}"
            
            Steps:
            1. Identify general principles or rules that apply
            2. Apply these rules to the specific situation
            3. Draw logical conclusions
            
            Provide your reasoning step by step.
            """
            
            try:
                response = await self.groq_client.chat_completion([{"role": "user", "content": prompt}])
                return {
                    'reasoning_type': 'deductive',
                    'reasoning_steps': response,
                    'confidence': 0.85,
                    'conclusion': self._extract_conclusion(response)
                }
            except Exception as e:
                self.logger.error(f"Deductive reasoning with LLM failed: {str(e)}")
        
        # Fallback to simple rule-based reasoning
        return {
            'reasoning_type': 'deductive',
            'reasoning_steps': ["Applied general knowledge rules to specific case"],
            'confidence': 0.7,
            'conclusion': "Based on general principles, this seems reasonable."
        }

    async def _inductive_reasoning(self, message: str, analysis: Dict[str, Any], context: List[Dict]) -> Dict[str, Any]:
        """Apply inductive reasoning (specific to general)"""
        return {
            'reasoning_type': 'inductive',
            'reasoning_steps': ["Gathered specific observations", "Identified patterns", "Formed generalization"],
            'confidence': 0.6,
            'conclusion': "Based on the available evidence, this pattern appears consistent."
        }

    async def _abductive_reasoning(self, message: str, analysis: Dict[str, Any], context: List[Dict]) -> Dict[str, Any]:
        """Apply abductive reasoning (inference to best explanation)"""
        return {
            'reasoning_type': 'abductive',
            'reasoning_steps': ["Considered multiple explanations", "Evaluated evidence for each", "Selected most plausible"],
            'confidence': 0.75,
            'conclusion': "This explanation best fits the available facts."
        }

    async def _analogical_reasoning(self, message: str, analysis: Dict[str, Any], context: List[Dict]) -> Dict[str, Any]:
        """Apply analogical reasoning (comparison and analogy)"""
        return {
            'reasoning_type': 'analogical',
            'reasoning_steps': ["Identied similar situations", "Mapped correspondences", "Applied insights"],
            'confidence': 0.65,
            'conclusion': "This is similar to other cases I've encountered."
        }

    async def _practical_reasoning(self, message: str, analysis: Dict[str, Any], context: List[Dict]) -> Dict[str, Any]:
        """Apply practical reasoning (pragmatic problem-solving)"""
        return {
            'reasoning_type': 'practical',
            'reasoning_steps': ["Defined the practical problem", "Considered available options", "Selected most effective solution"],
            'confidence': 0.8,
            'conclusion': "This approach should work effectively in practice."
        }

    async def _generate_response(self, reasoning_result: Dict[str, Any], analysis: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Generate final response based on reasoning"""
        # Determine response tone based on analysis
        tone = self._determine_response_tone(analysis, reasoning_result)
        
        # Use LLM for response generation if available
        if self.groq_client:
            try:
                prompt = self._build_response_prompt(reasoning_result, analysis, tone)
                response = await self.groq_client.chat_completion([{"role": "user", "content": prompt}])
                
                return {
                    'text': response,
                    'tone': tone,
                    'generation_method': 'llm'
                }
            except Exception as e:
                self.logger.error(f"LLM response generation failed: {str(e)}")
        
        # Fallback to template-based response
        return {
            'text': self._generate_template_response(reasoning_result, analysis, tone),
            'tone': tone,
            'generation_method': 'template'
        }

    def _determine_response_tone(self, analysis: Dict[str, Any], reasoning_result: Dict[str, Any]) -> str:
        """Determine appropriate response tone"""
        sentiment = analysis['sentiment']
        intent = analysis['intent']
        
        if intent == 'humor' or random.random() < 0.3:  # 30% chance for humor
            return 'humorous'
        elif sentiment == 'positive':
            return 'creative'
        elif analysis['complexity'] == 'complex':
            return 'analytical'
        else:
            return 'practical'

    def _build_response_prompt(self, reasoning_result: Dict[str, Any], analysis: Dict[str, Any], tone: str) -> str:
        """Build prompt for LLM response generation"""
        base_prompt = f"""
        Generate a response with {tone} tone based on this reasoning:
        
        User Message: {analysis.get('original_message', 'Unknown')}
        Reasoning Type: {reasoning_result.get('reasoning_type', 'unknown')}
        Reasoning Steps: {reasoning_result.get('reasoning_steps', [])}
        Conclusion: {reasoning_result.get('conclusion', 'No conclusion')}
        
        Additional Context:
        - Intent: {analysis.get('intent', 'unknown')}
        - Sentiment: {analysis.get('sentiment', 'neutral')}
        - Domain: {analysis.get('domain', 'general')}
        
        Please respond in a {tone} manner while being helpful and accurate.
        """
        
        return base_prompt

    def _generate_template_response(self, reasoning_result: Dict[str, Any], analysis: Dict[str, Any], tone: str) -> str:
        """Generate response using templates"""
        templates = {
            'analytical': "Based on careful analysis using {reasoning_type} reasoning, {conclusion}",
            'creative': "Here's an interesting perspective! {conclusion}",
            'practical': "The most practical approach here is: {conclusion}",
            'humorous': "Mickey's been thinking about this! {conclusion} And that's no joke! 😄"
        }
        
        template = templates.get(tone, templates['practical'])
        return template.format(
            reasoning_type=reasoning_result.get('reasoning_type', 'careful'),
            conclusion=reasoning_result.get('conclusion', 'I have some insights to share.')
        )

    def _extract_conclusion(self, reasoning_text: str) -> str:
        """Extract conclusion from reasoning text"""
        # Simple extraction - look for conclusion indicators
        conclusion_indicators = ['therefore', 'thus', 'so', 'conclusion', 'in summary']
        
        sentences = reasoning_text.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in conclusion_indicators):
                return sentence.strip()
        
        return reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text

    def _add_reasoning_step(self, step_type: str, description: str, data: Any = None):
        """Add a step to the reasoning chain"""
        step = ReasoningStep(step_type, description, data)
        if self.reasoning_chain:
            previous_step = self.reasoning_chain[-1]
            previous_step.duration = step.timestamp - previous_step.timestamp
        
        self.reasoning_chain.append(step)

    def _compile_reasoning_report(self, total_time: float) -> Dict[str, Any]:
        """Compile a report of the reasoning process"""
        if not self.reasoning_chain:
            return {}
        
        steps_summary = [
            {
                'step': step.step_type,
                'description': step.description,
                'duration': step.duration,
                'timestamp': step.timestamp
            }
            for step in self.reasoning_chain
        ]
        
        return {
            'total_steps': len(self.reasoning_chain),
            'total_time': total_time,
            'average_step_time': total_time / len(self.reasoning_chain) if self.reasoning_chain else 0,
            'steps': steps_summary,
            'final_state': self.current_state.value
        }

    def _get_reasoning_personality(self, tone: str) -> str:
        """Get Mickey's reasoning personality message"""
        return self.reasoning_personalities.get(tone, "Mickey's thinking about this! 🤔")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'response': "I encountered an error while processing that. Could you try again?",
            'mickey_response': "Oops! Mickey's brain had a little glitch! 🐭💥"
        }

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            'current_state': self.current_state.value,
            'reasoning_patterns_count': len(self.reasoning_patterns),
            'supported_domains': self.supported_domains,
            'cognitive_parameters': {
                'creativity_level': self.creativity_level,
                'analytical_depth': self.analytical_depth,
                'context_sensitivity': self.context_sensitivity
            }
        }

    def update_cognitive_parameters(self, creativity: float = None, analytical: float = None, context: float = None):
        """Update cognitive parameters"""
        if creativity is not None and 0 <= creativity <= 1:
            self.creativity_level = creativity
        if analytical is not None and 0 <= analytical <= 1:
            self.analytical_depth = analytical
        if context is not None and 0 <= context <= 1:
            self.context_sensitivity = context
        
        self.logger.info(f"Updated cognitive parameters: creativity={self.creativity_level}, "
                        f"analytical={self.analytical_depth}, context={self.context_sensitivity}")

# Test function
async def test_reasoning_engine():
    """Test the reasoning engine"""
    reasoning_engine = ReasoningEngine()
    
    test_messages = [
        "What's the difference between artificial intelligence and machine learning?",
        "Tell me a funny story about technology",
        "How should I approach learning programming?",
        "Compare traditional education with online learning"
    ]
    
    for message in test_messages:
        print(f"\nTesting: '{message}'")
        result = await reasoning_engine.process_message(message)
        print(f"Response: {result['response']}")
        print(f"Reasoning Approach: {result['reasoning_approach']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
    
    # Test stats
    stats = reasoning_engine.get_reasoning_stats()
    print(f"\nReasoning Stats: {stats}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_reasoning_engine())