"""
Evaluation Framework for Legal AI Assistant
Metrics: Accuracy, Hallucination Rate, Relevance, Expert Validation
"""

import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from collections import defaultdict
import json

class LegalRAGEvaluator:
    """
    Comprehensive evaluation framework for Legal RAG system
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.evaluation_results = defaultdict(list)
    
    def calculate_retrieval_recall(self, 
                                   retrieved_docs: List[Dict], 
                                   relevant_docs: List[Dict],
                                   k: int = 3) -> float:
        """
        Calculate Recall@K for retrieval
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Ground truth relevant documents
            k: Number of top documents to consider
        
        Returns:
            Recall@K score
        """
        retrieved_ids = set(doc.get('id', doc.get('citation', '')) for doc in retrieved_docs[:k])
        relevant_ids = set(doc.get('id', doc.get('citation', '')) for doc in relevant_docs)
        
        if not relevant_ids:
            return 0.0
        
        intersection = retrieved_ids.intersection(relevant_ids)
        recall = len(intersection) / len(relevant_ids)
        
        return recall
    
    def calculate_precision(self,
                           retrieved_docs: List[Dict],
                           relevant_docs: List[Dict],
                           k: int = 3) -> float:
        """
        Calculate Precision@K for retrieval
        """
        retrieved_ids = set(doc.get('id', doc.get('citation', '')) for doc in retrieved_docs[:k])
        relevant_ids = set(doc.get('id', doc.get('citation', '')) for doc in relevant_docs)
        
        if not retrieved_ids:
            return 0.0
        
        intersection = retrieved_ids.intersection(relevant_ids)
        precision = len(intersection) / len(retrieved_ids)
        
        return precision
    
    def calculate_rouge_scores(self, 
                               generated_text: str, 
                               reference_text: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for generated vs reference text
        
        Args:
            generated_text: AI-generated response
            reference_text: Ground truth reference
        
        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        scores = self.rouge_scorer.score(reference_text, generated_text)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def detect_hallucinations(self,
                             generated_text: str,
                             retrieved_sources: List[Dict]) -> Dict[str, any]:
        """
        Detect potential hallucinations in generated text
        
        Checks if citations mentioned exist in retrieved sources
        
        Args:
            generated_text: AI-generated response
            retrieved_sources: Documents that were retrieved
        
        Returns:
            Dict with hallucination metrics
        """
        # Extract citation patterns (e.g., [1], [2020] UKSC 15)
        import re
        citation_pattern = r'\[[\d]+\]|\[\d{4}\]\s+[A-Z]+\s+\d+'
        mentioned_citations = set(re.findall(citation_pattern, generated_text))
        
        # Get available citations from sources
        available_citations = set()
        for source in retrieved_sources:
            available_citations.add(source.get('citation', ''))
            # Also check for bracketed numbers
            source_text = source.get('text', '')
            available_citations.update(re.findall(citation_pattern, source_text))
        
        # Check for unsupported citations
        unsupported = mentioned_citations - available_citations
        
        hallucination_rate = len(unsupported) / max(len(mentioned_citations), 1)
        
        return {
            'hallucination_rate': hallucination_rate,
            'total_citations': len(mentioned_citations),
            'unsupported_citations': len(unsupported),
            'unsupported_list': list(unsupported)
        }
    
    def calculate_relevance_score(self,
                                 query: str,
                                 response: str,
                                 retrieved_docs: List[Dict]) -> float:
        """
        Calculate semantic relevance between query and response
        Uses keyword overlap as simple relevance metric
        """
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are'}
        query_words = query_words - stop_words
        response_words = response_words - stop_words
        
        if not query_words:
            return 0.0
        
        overlap = query_words.intersection(response_words)
        relevance = len(overlap) / len(query_words)
        
        return relevance
    
    def evaluate_single_query(self,
                             query: str,
                             generated_response: str,
                             retrieved_docs: List[Dict],
                             ground_truth: Dict = None) -> Dict:
        """
        Comprehensive evaluation for a single query
        
        Args:
            query: User query
            generated_response: System's generated response
            retrieved_docs: Documents retrieved by RAG
            ground_truth: Optional ground truth data with 'answer' and 'relevant_docs'
        
        Returns:
            Dict with all evaluation metrics
        """
        metrics = {}
        
        # Hallucination detection
        hallucination_metrics = self.detect_hallucinations(generated_response, retrieved_docs)
        metrics.update(hallucination_metrics)
        
        # Relevance score
        metrics['relevance_score'] = self.calculate_relevance_score(
            query, generated_response, retrieved_docs
        )
        
        # Number of sources cited
        metrics['sources_cited'] = len(retrieved_docs)
        
        # If ground truth available, calculate additional metrics
        if ground_truth:
            if 'answer' in ground_truth:
                rouge_scores = self.calculate_rouge_scores(
                    generated_response, ground_truth['answer']
                )
                metrics.update(rouge_scores)
            
            if 'relevant_docs' in ground_truth:
                metrics['recall@3'] = self.calculate_retrieval_recall(
                    retrieved_docs, ground_truth['relevant_docs'], k=3
                )
                metrics['precision@3'] = self.calculate_precision(
                    retrieved_docs, ground_truth['relevant_docs'], k=3
                )
        
        return metrics
    
    def evaluate_rag_vs_baseline(self,
                                 test_queries: List[Dict],
                                 rag_pipeline,
                                 baseline_model=None) -> Dict:
        """
        Compare RAG system against baseline LLM
        
        Args:
            test_queries: List of dicts with 'query', 'ground_truth' keys
            rag_pipeline: RAG pipeline instance
            baseline_model: Optional baseline model for comparison
        
        Returns:
            Comparative evaluation results
        """
        results = {
            'rag': [],
            'baseline': [],
            'comparison': {}
        }
        
        for test in test_queries:
            query = test['query']
            ground_truth = test.get('ground_truth')
            
            # Evaluate RAG
            rag_response = rag_pipeline.query(query)
            rag_metrics = self.evaluate_single_query(
                query,
                rag_response['answer'],
                rag_response['sources'],
                ground_truth
            )
            results['rag'].append(rag_metrics)
            
            # Evaluate baseline if provided
            if baseline_model:
                baseline_response = baseline_model.generate(query)
                baseline_metrics = self.evaluate_single_query(
                    query,
                    baseline_response,
                    [],  # No retrieval in baseline
                    ground_truth
                )
                results['baseline'].append(baseline_metrics)
        
        # Calculate aggregate statistics
        results['comparison'] = self._calculate_aggregate_stats(results)
        
        return results
    
    def _calculate_aggregate_stats(self, results: Dict) -> Dict:
        """Calculate mean and std for all metrics"""
        stats = {}
        
        for mode in ['rag', 'baseline']:
            if mode in results and results[mode]:
                mode_stats = {}
                
                # Get all metric keys
                all_keys = set()
                for result in results[mode]:
                    all_keys.update(result.keys())
                
                # Calculate stats for each metric
                for key in all_keys:
                    values = [r[key] for r in results[mode] if key in r]
                    if values and isinstance(values[0], (int, float)):
                        mode_stats[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                stats[mode] = mode_stats
        
        return stats
    
    def generate_evaluation_report(self, 
                                  results: Dict,
                                  output_file: str = 'evaluation_report.json'):
        """
        Generate comprehensive evaluation report
        """
        report = {
            'summary': results['comparison'],
            'detailed_results': {
                'rag': results['rag'],
                'baseline': results.get('baseline', [])
            },
            'key_findings': self._extract_key_findings(results)
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {output_file}")
        
        return report
    
    def _extract_key_findings(self, results: Dict) -> Dict:
        """Extract key findings from evaluation"""
        findings = {}
        
        if 'rag' in results['comparison']:
            rag_stats = results['comparison']['rag']
            
            findings['average_hallucination_rate'] = rag_stats.get('hallucination_rate', {}).get('mean', 0)
            findings['average_relevance'] = rag_stats.get('relevance_score', {}).get('mean', 0)
            findings['average_sources_cited'] = rag_stats.get('sources_cited', {}).get('mean', 0)
            
            if 'recall@3' in rag_stats:
                findings['average_recall'] = rag_stats['recall@3'].get('mean', 0)
        
        return findings


# Example usage
if __name__ == "__main__":
    # Example test cases
    test_queries = [
        {
            'query': 'What are the rights regarding unfair dismissal?',
            'ground_truth': {
                'answer': 'Employees have the right not to be unfairly dismissed under ERA 1996.',
                'relevant_docs': [
                    {'id': 1, 'citation': 'ERA 1996 s.94'}
                ]
            }
        },
        {
            'query': 'What is the legal status of Uber drivers?',
            'ground_truth': {
                'answer': 'Uber drivers are classified as workers under UK law.',
                'relevant_docs': [
                    {'id': 2, 'citation': '[2020] UKSC 15'}
                ]
            }
        }
    ]
    
    # Initialize evaluator
    evaluator = LegalRAGEvaluator()
    
    # Example evaluation
    print("Evaluation Framework Initialized")
    print(f"Test Queries: {len(test_queries)}")
    
    # Example single query evaluation
    example_response = "Under the Employment Rights Act 1996 [1], employees have the right not to be unfairly dismissed."
    example_sources = [
        {'citation': 'ERA 1996 s.94', 'text': 'Right not to be unfairly dismissed...'}
    ]
    
    metrics = evaluator.evaluate_single_query(
        "What are the rights regarding unfair dismissal?",
        example_response,
        example_sources
    )
    
    print("\nExample Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
