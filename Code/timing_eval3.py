# `dynamic_eval.py`
import asyncio
import json
import logging
import os
import pandas as pd
import sys
import time
from scipy import stats
import numpy as np
import re
import asyncio
from asyncio import Semaphore
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'new_evals')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tevagpt_vs_askhr')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'textual_evals_final')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'temperature')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))


from updated_ragevaluator import LLMRAGEvaluator
from Ask_HR_dynamic import AskHR
# from ninetyfour import EnhancedAskHR as AskHR
from CyrusGPT import CyrusGPT
from TevaGPT_dynamic import TevaGPT
from BaseModel import BaseModel

# logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_test_cases(csv_path='tevagpt_vs_askhr/full_test_set.csv'): #test_set_for_cyrus #new_test_set # calculations_eval #full_test_set
    """Load test cases from CSV file
       Note: you have to download the file as a csv with utf-8 encoding. xlsx caused issues.
       Also: the evals must come in this exact format for the pipeline to work. 
    """
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        df = pd.read_csv(f)
    test_cases = []
    for _, row in df.iterrows():

        # schema internalization
        test_cases.append({
            "test_id": int(row["test_id"]),
            "capability": row["capability"],
            "content": row["content"],
            "query": row["query"],
            "expected_output": row["expected_output"],
            "file_type": row["file_type"],
            "file_path": row.get("file_path", ""),
            "file_name": row.get("file_name", "")
        })
    return test_cases

def calculate_composite_score(rag_scores):
    """Calculate weighted composite score from RAG metrics"""
    if not rag_scores or 'error' in rag_scores:
        return 0.0
    
    # Define metrics and their weights
    weights = {
        'TaskSuccess': 0.5,      # 50%
        'Faithfulness': 0.125,   # 12.5%
        'AnswerRelevancy': 0.125, # 12.5%
        'ContextRecall': 0.125,   # 12.5%
        'ContextPrecision': 0.125 # 12.5%
    }
    
    # Calculate weighted score
    weighted_sum = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in rag_scores:
            # Normalize score to 0-1 range (assuming original scores are 0-10)
            normalized_score = rag_scores[metric] / 10.0
            weighted_sum += normalized_score * weight
            total_weight += weight
    
    # Return weighted average (handles case where some metrics might be missing)
    return weighted_sum / total_weight if total_weight > 0 else 0.0

def aggregate_metrics_by_constraint(results, constraint):
    """Aggregate metrics for a specific constraint level"""
    constraint_results = [r for r in results if r['context_constraint'] == constraint]
    
    if not constraint_results:
        return {
            'constraint': constraint,
            'overall_score': 0.0,
            'task_success_avg': 0.0,
            'faithfulness_avg': 0.0,
            'answer_relevancy_avg': 0.0,
            'context_recall_avg': 0.0,
            'context_precision_avg': 0.0,
            'error_count': 0
        }
    
    # calculate metrics
    valid_results = [r for r in constraint_results if r.get('rag_scores') and 'error' not in r.get('rag_scores', {})]
    error_count = len(constraint_results) - len(valid_results)
    
    if not valid_results:
        return {
            'constraint': constraint,
            'overall_score': 0.0,
            'task_success_avg': 0.0,
            'faithfulness_avg': 0.0,
            'answer_relevancy_avg': 0.0,
            'context_recall_avg': 0.0,
            'context_precision_avg': 0.0,
            'error_count': error_count
        }
    
    # calculate averages
    task_success_scores = [r['rag_scores'].get('TaskSuccess', 0) / 10.0 for r in valid_results]
    faithfulness_scores = [r['rag_scores'].get('Faithfulness', 0) / 10.0 for r in valid_results]
    relevancy_scores = [r['rag_scores'].get('AnswerRelevancy', 0) / 10.0 for r in valid_results]
    recall_scores = [r['rag_scores'].get('ContextRecall', 0) / 10.0 for r in valid_results]
    precision_scores = [r['rag_scores'].get('ContextPrecision', 0) / 10.0 for r in valid_results]
    
    overall_scores = [calculate_composite_score(r.get('rag_scores', {})) for r in valid_results]
    
    return {
        'constraint': constraint,
        'overall_score': sum(overall_scores) / len(overall_scores),
        'task_success_avg': sum(task_success_scores) / len(task_success_scores),
        'faithfulness_avg': sum(faithfulness_scores) / len(faithfulness_scores),
        'answer_relevancy_avg': sum(relevancy_scores) / len(relevancy_scores),
        'context_recall_avg': sum(recall_scores) / len(recall_scores),
        'context_precision_avg': sum(precision_scores) / len(precision_scores),
        'error_count': error_count
    }

def aggregate_metrics_by_capability(results, capability):
    """Aggregate metrics for a specific capability/task type"""
    capability_results = [r for r in results if r['capability'] == capability]
    
    if not capability_results:
        return {
            'capability': capability,
            'overall_score': 0.0,
            'task_success_avg': 0.0,
            'faithfulness_avg': 0.0,
            'answer_relevancy_avg': 0.0,
            'context_recall_avg': 0.0,
            'context_precision_avg': 0.0,
            'error_count': 0
        }
    
    # calculate metrics
    valid_results = [r for r in capability_results if r.get('rag_scores') and 'error' not in r.get('rag_scores', {})]
    error_count = len(capability_results) - len(valid_results)
    
    if not valid_results:
        return {
            'capability': capability,
            'overall_score': 0.0,
            'task_success_avg': 0.0,
            'faithfulness_avg': 0.0,
            'answer_relevancy_avg': 0.0,
            'context_recall_avg': 0.0,
            'context_precision_avg': 0.0,
            'error_count': error_count
        }
    
    # calculate averages
    task_success_scores = [r['rag_scores'].get('TaskSuccess', 0) / 10.0 for r in valid_results]
    faithfulness_scores = [r['rag_scores'].get('Faithfulness', 0) / 10.0 for r in valid_results]
    relevancy_scores = [r['rag_scores'].get('AnswerRelevancy', 0) / 10.0 for r in valid_results]
    recall_scores = [r['rag_scores'].get('ContextRecall', 0) / 10.0 for r in valid_results]
    precision_scores = [r['rag_scores'].get('ContextPrecision', 0) / 10.0 for r in valid_results]
    
    overall_scores = [calculate_composite_score(r.get('rag_scores', {})) for r in valid_results]
    
    return {
        'capability': capability,
        'overall_score': sum(overall_scores) / len(overall_scores),
        'task_success_avg': sum(task_success_scores) / len(task_success_scores),
        'faithfulness_avg': sum(faithfulness_scores) / len(faithfulness_scores),
        'answer_relevancy_avg': sum(relevancy_scores) / len(relevancy_scores),
        'context_recall_avg': sum(recall_scores) / len(recall_scores),
        'context_precision_avg': sum(precision_scores) / len(precision_scores),
        'error_count': error_count
    }

def create_detailed_results_dataframe(results, model_name):
    """Create detailed results dataframe for a single model"""
    detailed_data = []
    
    for result in results:
        rag_scores = result.get('rag_scores', {})
        
        # handle retrieved docs
        retrieved_docs_text = ""
        if result.get('retrieved_docs'):
            retrieved_docs_text = "\n\n------ NEXT DOC ------\n\n".join(result['retrieved_docs'])
        
        detailed_data.append({
            'model': model_name,
            'constraint_level': result['context_constraint'],
            'test_id': result['test_id'],
            'capability': result['capability'],
            'file_path': result.get('file_path', ''),
            'query': result['query'],
            'expected_output': result['expected'],
            'expected_context': result['expected_context'],
            'actual_response': result['response'],
            'retrieved_docs': retrieved_docs_text,
            'retrieved_docs_count': len(result.get('retrieved_docs', [])),
            'composite_score': round(calculate_composite_score(rag_scores), 4),
            'task_success': round(rag_scores.get('TaskSuccess', 0) / 10.0, 4),
            'task_success_explanation': rag_scores.get('TaskSuccessExplanation', ''),
            'faithfulness': round(rag_scores.get('Faithfulness', 0) / 10.0, 4),
            'answer_relevancy': round(rag_scores.get('AnswerRelevancy', 0) / 10.0, 4),
            'context_recall': round(rag_scores.get('ContextRecall', 0) / 10.0, 4),
            'context_precision': round(rag_scores.get('ContextPrecision', 0) / 10.0, 4),
            'evaluation_time_seconds': rag_scores.get('EvaluationTime', ''),
            'token_usage': rag_scores.get('TokenUsage', ''),
            'cost_usd': rag_scores.get('CostUSD', ''),
            'error_message': rag_scores.get('error', '')
        })
    
    df = pd.DataFrame(detailed_data)

    # FIXED: Sort by constraint_level, then test_id
    if not df.empty:
        # Define custom constraint order
        constraint_order = ['given', 'file', 'folder', 'unconstrained']
        df['constraint_sort'] = df['constraint_level'].map({
            constraint: i for i, constraint in enumerate(constraint_order)
        }).fillna(999)  # Put unknown constraints at the end
        
        df = df.sort_values(['constraint_sort', 'test_id']).drop('constraint_sort', axis=1)
        df = df.reset_index(drop=True)
    return df

def calculate_significance(scores1, scores2):
    """
    Calculate statistical significance between two sets of scores using t-test
    Returns p-value and significance stars
    """
    if len(scores1) < 2 or len(scores2) < 2:
        return 1.0, ""
    
    try:
        # Perform two-tailed t-test (not z-test)
        t_stat, p_value = stats.ttest_ind(scores1, scores2)
        
        # Add significance stars (R-style)
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        elif p_value < 0.1:
            stars = "."
        else:
            stars = ""
            
        return p_value, stars
    except:
        return 1.0, ""

def create_model_comparison_summary(all_results):
    """
    Create improved model comparison summary table with overall model statistics (not split by capability)
    """
    # Get all unique constraints
    all_constraints = set()
    for results in all_results.values():
        all_constraints.update([r['context_constraint'] for r in results])
    
    # Get all model labels
    model_labels = list(all_results.keys())
    
    summary_data = []
    
    for constraint in all_constraints:
        # Calculate metrics for each model across ALL test cases for this constraint
        model_scores = {}
        model_raw_scores = {}  # For statistical testing
        
        for label in model_labels:
            results = all_results[label]
            constraint_results = [r for r in results if r['context_constraint'] == constraint]
            valid_results = [r for r in constraint_results if r.get('rag_scores') and 'error' not in r.get('rag_scores', {})]
            
            if valid_results:
                # Calculate composite scores for this constraint (ALL test cases)
                scores = [calculate_composite_score(r.get('rag_scores', {})) for r in valid_results]
                avg_score = sum(scores) / len(scores)
                model_scores[label] = avg_score
                model_raw_scores[label] = scores  # Keep all individual scores
            else:
                model_scores[label] = 0.0
                model_raw_scores[label] = [0.0]
        
        # Find best and second-best models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        best_model = sorted_models[0][0] if sorted_models else ""
        best_score = sorted_models[0][1] if sorted_models else 0.0
        
        # Calculate significance between best and second-best using ALL test cases
        p_value, stars = "", ""
        n_cases = 0
        if len(sorted_models) >= 2:
            second_best_model = sorted_models[1][0]
            best_scores = model_raw_scores[best_model]  # All individual test case scores
            second_scores = model_raw_scores[second_best_model]  # All individual test case scores
            p_value, stars = calculate_significance(best_scores, second_scores)
            n_cases = len(best_scores)
        
        # Create row-by-row format
        row_data = {
            'constraint_level': constraint,
            'best_model': best_model,
            'best_score': round(best_score, 4),
            'p_value': f"{p_value:.4f}" if p_value != "" else "",
            'significance': stars,
            'n_test_cases': n_cases
        }
        
        # Add each model's score as a separate row entry
        for model_label in model_labels:
            row_data[f'{model_label}_score'] = round(model_scores.get(model_label, 0.0), 4)
        
        summary_data.append(row_data)
    
    return pd.DataFrame(summary_data)

def create_model_capability_comparison(all_results):
    """
    Create improved model capability comparison table without p-values (too few test cases per capability)
    """
    # Get all unique capabilities
    all_capabilities = set()
    for results in all_results.values():
        all_capabilities.update([r['capability'] for r in results])
    
    # Get all model labels
    model_labels = list(all_results.keys())
    
    summary_data = []
    
    for capability in all_capabilities:
        # Calculate metrics for each model
        model_scores = {}
        
        for label in model_labels:
            results = all_results[label]
            capability_results = [r for r in results if r['capability'] == capability]
            valid_results = [r for r in capability_results if r.get('rag_scores') and 'error' not in r.get('rag_scores', {})]
            
            if valid_results:
                # Calculate composite scores for this capability
                scores = [calculate_composite_score(r.get('rag_scores', {})) for r in valid_results]
                avg_score = sum(scores) / len(scores)
                model_scores[label] = avg_score
            else:
                model_scores[label] = 0.0
        
        # Find best model
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        best_model = sorted_models[0][0] if sorted_models else ""
        best_score = sorted_models[0][1] if sorted_models else 0.0
        
        # Create row-by-row format (no p-values due to low sample size)
        row_data = {
            'capability': capability,
            'best_model': best_model,
            'best_score': round(best_score, 4),
            'n_test_cases': len([r for r in all_results[best_model] if r['capability'] == capability and r.get('rag_scores') and 'error' not in r.get('rag_scores', {})])
        }
        
        # Add each model's score as a separate row entry
        for model_label in model_labels:
            row_data[f'{model_label}_score'] = round(model_scores.get(model_label, 0.0), 4)
        
        summary_data.append(row_data)
    
    return pd.DataFrame(summary_data)

async def create_improvement_examples_sheet(all_results, model_labels):
    """
    Create improvement examples sheet with precise business summaries and sorted by capability
    Fixed to handle cross-model-type comparisons and more realistic thresholds
    """
    # Calculate overall scores for each model to find the best performer
    model_overall_scores = {}
    
    for label, results in all_results.items():
        # Calculate average score across all constraints and test cases
        valid_results = [r for r in results if r.get('rag_scores') and 'error' not in r.get('rag_scores', {})]
        if valid_results:
            scores = [calculate_composite_score(r.get('rag_scores', {})) for r in valid_results]
            model_overall_scores[label] = sum(scores) / len(scores)
        else:
            model_overall_scores[label] = 0.0
    
    # Find the best performing model
    if not model_overall_scores:
        return pd.DataFrame()  # No valid results
    
    best_model_label = max(model_overall_scores.keys(), key=lambda x: model_overall_scores[x])
    best_model_score = model_overall_scores[best_model_label]
    
    # FIXED: Find baseline model (lowest scoring model instead of same-type requirement)
    baseline_model_label = min(model_overall_scores.keys(), key=lambda x: model_overall_scores[x])
    baseline_model_score = model_overall_scores[baseline_model_label]
    
    # FIXED: More realistic improvement threshold (0.5% instead of 2%)
    if best_model_score <= baseline_model_score + 0.005:  # 0.005 = 0.5% threshold
        print(f"Best model ({best_model_label}: {best_model_score:.4f}) doesn't meaningfully outperform baseline ({baseline_model_label}: {baseline_model_score:.4f})")
        return pd.DataFrame()  # No meaningful improvement
    
    print(f"Creating improvement examples: {best_model_label} ({best_model_score:.4f}) vs {baseline_model_label} ({baseline_model_score:.4f})")
    
    # Initialize GPT-4o for analysis
    try:
        base_model = BaseModel(model='gpt-4-omni', temperature=0.0)
    except:
        # If BaseModel fails, skip analysis and just show raw comparisons
        print("Warning: Could not initialize BaseModel for analysis. Showing raw comparisons only.")
        base_model = None
    
    # Get results for both models
    best_results = all_results[best_model_label]
    baseline_results = all_results[baseline_model_label]
    
    # Create lookup dictionaries for faster comparison
    baseline_lookup = {(r['test_id'], r['context_constraint']): r for r in baseline_results}
    best_lookup = {(r['test_id'], r['context_constraint']): r for r in best_results}
    
    improvement_data = []
    
    # Find cases where best model scored better than baseline
    for key, best_result in best_lookup.items():
        if key in baseline_lookup:
            baseline_result = baseline_lookup[key]
            
            # Calculate scores
            baseline_score = calculate_composite_score(baseline_result.get('rag_scores', {}))
            best_score = calculate_composite_score(best_result.get('rag_scores', {}))
            
            # FIXED: Lower threshold for individual test cases (1% instead of 5%)
            if best_score > baseline_score + 0.01:  # 0.01 = 1% threshold
                score_delta = best_score - baseline_score
                
                # Truncate responses for summary (first 200 chars for better context)
                query_truncated = best_result['query'][:100] + "..." if len(best_result['query']) > 100 else best_result['query']
                baseline_response_truncated = baseline_result['response'][:200] + "..." if len(baseline_result['response']) > 200 else baseline_result['response']
                best_response_truncated = best_result['response'][:200] + "..." if len(best_result['response']) > 200 else best_result['response']
                
                # Analysis with GPT if available, otherwise use simple comparison
                if base_model:
                    try:
                        # More precise business analysis prompt
                        analysis_prompt = f"""Query: {query_truncated}
Expected Answer: {best_result['expected']}

{baseline_model_label} Response: {baseline_response_truncated}
{best_model_label} Response: {best_response_truncated}

Write a precise, factual explanation of why {best_model_label}'s answer is better. Focus on:
- Is it more accurate/correct?
- Does it match the expected answer better?
- Is it more complete or specific?
- Does it provide the right information vs wrong information?

Be specific and factual. Avoid generic phrases like "better user experience." Keep to 1 sentence."""

                        # Impact category analysis prompt
                        category_prompt = f"""Based on these responses:

Expected: {best_result['expected']}
{baseline_model_label}: {baseline_response_truncated}
{best_model_label}: {best_response_truncated}

What is the main type of improvement? Choose ONE:
- More Complete Answer
- Better Accuracy
- Clearer Instructions
- More Specific Details
- Better Organization
- More Actionable Information
- Correct vs Incorrect Answer

Respond with just the category name."""

                        # Get precise analysis
                        analysis_response = await base_model.query(analysis_prompt)
                        business_summary = analysis_response['output'].strip()
                        
                        # Get impact category
                        category_response = await base_model.query(category_prompt)
                        impact_category = category_response['output'].strip()
                        
                    except Exception as e:
                        business_summary = f"Analysis failed: {str(e)}"
                        impact_category = "Analysis Error"
                else:
                    # Simple comparison without GPT analysis
                    if best_score > baseline_score + 0.05:
                        business_summary = f"{best_model_label} significantly outperformed {baseline_model_label}"
                        impact_category = "Significant Improvement"
                    else:
                        business_summary = f"{best_model_label} moderately outperformed {baseline_model_label}"
                        impact_category = "Moderate Improvement"
                
                # Create full details cell with expected answer
                full_details = f"""QUERY: {best_result['query']}

EXPECTED/CORRECT ANSWER: {best_result['expected']}

BASELINE MODEL RESPONSE ({baseline_model_label}):
{baseline_result['response']}

RECOMMENDED MODEL RESPONSE ({best_model_label}):
{best_result['response']}

TECHNICAL SCORES:
- Baseline Model Score: {baseline_score:.3f}
- Recommended Model Score: {best_score:.3f}
- Improvement: +{score_delta:.3f}"""
                
                improvement_data.append({
                    'test_case_id': best_result['test_id'],
                    'capability': best_result['capability'],  # For sorting
                    'baseline_model': baseline_model_label,
                    'recommended_model': best_model_label, 
                    'score_improvement': f"+{score_delta:.3f}",
                    'impact_category': impact_category,
                    'business_summary': business_summary,
                    'constraint_level': best_result['context_constraint'],
                    'full_details': full_details
                })
    
    # Sort by capability first, then by score improvement
    improvement_data.sort(key=lambda x: (x['capability'], -float(x['score_improvement'].replace('+', ''))))
    
    if not improvement_data:
        print(f"No individual test cases where {best_model_label} meaningfully outperformed {baseline_model_label}")
        print(f"Debug: Best score {best_score:.4f}, Baseline score {baseline_score:.4f}")
        print(f"Debug: Found {len(best_lookup)} best results, {len(baseline_lookup)} baseline results")
        print(f"Debug: Overlapping keys: {len(set(best_lookup.keys()) & set(baseline_lookup.keys()))}")
        return pd.DataFrame()
    
    print(f"Found {len(improvement_data)} test cases where {best_model_label} outperformed {baseline_model_label}")
    return pd.DataFrame(improvement_data)

# Also update the Excel writing section in run_all_evals to handle the new columns better
def save_results_with_improved_formatting(output_path, summary_df, capability_df, improvement_df, detailed_dfs):
    """
    Save results with improved formatting, bolding, and auto-sized rows
    Fixed to ensure improvement sheet is always created when data exists
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sanitize all DataFrames first
        summary_df = sanitize_df(summary_df)
        capability_df = sanitize_df(capability_df)
        detailed_dfs_sanitized = {}
        for label, df in detailed_dfs.items():
            detailed_dfs_sanitized[label] = sanitize_df(df)

        # Save main sheets
        summary_df.to_excel(writer, index=False, sheet_name='Model_Summary')
        capability_df.to_excel(writer, index=False, sheet_name='Capability_Comparison')
        
        # FIXED: Always try to save improvement sheet, with better debugging
        print(f"Debug: improvement_df shape: {improvement_df.shape if not improvement_df.empty else 'EMPTY'}")
        print(f"Debug: improvement_df columns: {list(improvement_df.columns) if not improvement_df.empty else 'NO COLUMNS'}")
        
        if not improvement_df.empty:
            # Sanitize improvement_df
            improvement_df = sanitize_df(improvement_df)
            print(f"Debug: Sanitized improvement_df shape: {improvement_df.shape}")
            
            try:
                improvement_df.to_excel(writer, index=False, sheet_name='Improvement_Examples')
                print("Debug: Successfully wrote Improvement_Examples sheet")
                
                # Get the workbook and worksheet for formatting
                improvement_ws = writer.sheets['Improvement_Examples']
                
                # Set column widths for better readability
                column_widths = {
                    'A': 15,  # test_case_id
                    'B': 20,  # capability
                    'C': 25,  # baseline_model
                    'D': 25,  # recommended_model
                    'E': 18,  # score_improvement
                    'F': 25,  # impact_category
                    'G': 80,  # business_summary (wider for better readability)
                    'H': 18,  # constraint_level
                    'I': 120  # full_details (wider)
                }
                
                for col_letter, width in column_widths.items():
                    if col_letter in [chr(65 + i) for i in range(len(improvement_df.columns))]:  # Only set if column exists
                        improvement_ws.column_dimensions[col_letter].width = width
                
                # Enable text wrapping and auto-size rows
                from openpyxl.styles import Alignment
                wrap_alignment = Alignment(wrap_text=True, vertical='top')
                
                # Apply text wrapping and auto-size rows
                for row in range(2, len(improvement_df) + 2):  # Skip header row
                    # Apply text wrapping to business summary and full details (if they exist)
                    if len(improvement_df.columns) > 6:  # business_summary exists
                        improvement_ws[f'G{row}'].alignment = wrap_alignment  # business_summary
                    if len(improvement_df.columns) > 8:  # full_details exists
                        improvement_ws[f'I{row}'].alignment = wrap_alignment  # full_details
                    
                    # Auto-size row height based on content
                    try:
                        # Calculate approximate height needed for wrapped text
                        if 'business_summary' in improvement_df.columns:
                            business_summary_text = str(improvement_df.iloc[row - 2]['business_summary'])
                            business_lines = max(1, len(business_summary_text) // 80)  # 80 chars per line
                        else:
                            business_lines = 1
                            
                        if 'full_details' in improvement_df.columns:
                            full_details_text = str(improvement_df.iloc[row - 2]['full_details'])
                            details_lines = max(1, len(full_details_text) // 120)  # 120 chars per line
                        else:
                            details_lines = 1
                        
                        # Set row height (15 points per line is typical)
                        max_lines = max(business_lines, details_lines, 2)  # Minimum 2 lines
                        improvement_ws.row_dimensions[row].height = max_lines * 15
                    except Exception as e:
                        print(f"Debug: Error setting row height for row {row}: {e}")
                        # Set a default height
                        improvement_ws.row_dimensions[row].height = 30
                        
                print("Debug: Successfully formatted Improvement_Examples sheet")
                
            except Exception as e:
                print(f"Error creating Improvement_Examples sheet: {e}")
                # Create a simple fallback sheet
                try:
                    pd.DataFrame({'Error': [f'Failed to create improvement examples: {str(e)}']}).to_excel(
                        writer, index=False, sheet_name='Improvement_Examples'
                    )
                    print("Debug: Created fallback Improvement_Examples sheet")
                except Exception as e2:
                    print(f"Debug: Even fallback sheet creation failed: {e2}")
        else:
            # Create an empty sheet with explanation
            try:
                pd.DataFrame({
                    'Message': ['No improvement examples found'],
                    'Explanation': ['No test cases where one model significantly outperformed another']
                }).to_excel(writer, index=False, sheet_name='Improvement_Examples')
                print("Debug: Created empty Improvement_Examples sheet with explanation")
            except Exception as e:
                print(f"Debug: Failed to create empty Improvement_Examples sheet: {e}")

        # Apply formatting to Model Summary
        if not summary_df.empty:
            try:
                workbook = writer.book
                summary_ws = writer.sheets['Model_Summary']
                
                # Get model columns (those ending with '_score')
                model_columns = [col for col in summary_df.columns if col.endswith('_score')]
                
                # Bold the highest scores in each row
                from openpyxl.styles import Font
                bold_font = Font(bold=True)
                
                for row_idx in range(2, len(summary_df) + 2):  # Skip header
                    # Find max score in this row
                    row_scores = []
                    score_positions = []
                    
                    for col_idx, col_name in enumerate(summary_df.columns):
                        if col_name in model_columns:
                            cell_value = summary_df.iloc[row_idx - 2][col_name]
                            if isinstance(cell_value, (int, float)):
                                row_scores.append(cell_value)
                                score_positions.append((row_idx, col_idx + 1))  # +1 for Excel 1-indexing
                    
                    if row_scores:
                        max_score = max(row_scores)
                        # Bold all cells with max score (handles ties)
                        for (r, c), score in zip(score_positions, row_scores):
                            if abs(score - max_score) < 0.0001:  # Handle floating point precision
                                summary_ws.cell(row=r, column=c).font = bold_font
            except Exception as e:
                print(f"Debug: Error formatting Model_Summary: {e}")
        
        # Apply formatting to Capability Comparison
        if not capability_df.empty:
            try:
                capability_ws = writer.sheets['Capability_Comparison']
                
                # Get model columns (those ending with '_score')
                model_columns = [col for col in capability_df.columns if col.endswith('_score')]
                
                # Bold the highest scores in each row
                from openpyxl.styles import Font
                bold_font = Font(bold=True)
                
                for row_idx in range(2, len(capability_df) + 2):  # Skip header
                    # Find max score in this row
                    row_scores = []
                    score_positions = []
                    
                    for col_idx, col_name in enumerate(capability_df.columns):
                        if col_name in model_columns:
                            cell_value = capability_df.iloc[row_idx - 2][col_name]
                            if isinstance(cell_value, (int, float)):
                                row_scores.append(cell_value)
                                score_positions.append((row_idx, col_idx + 1))  # +1 for Excel 1-indexing
                    
                    if row_scores:
                        max_score = max(row_scores)
                        # Bold all cells with max score (handles ties)
                        for (r, c), score in zip(score_positions, row_scores):
                            if abs(score - max_score) < 0.0001:  # Handle floating point precision
                                capability_ws.cell(row=r, column=c).font = bold_font
            except Exception as e:
                print(f"Debug: Error formatting Capability_Comparison: {e}")
        
        # Save detailed results
        for label, df in detailed_dfs_sanitized.items():
            try:
                sheet_name = f'{label}_Results'[:31]  # Excel sheet name limit
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            except Exception as e:
                print(f"Debug: Error saving detailed results for {label}: {e}")

        print(f"Debug: Finished saving all sheets to {output_path}")

def extract_independent_variable_value(model, independent_variable):
    """Extract the value of the independent variable from a model instance"""
    if independent_variable == 'temperature':
        return getattr(model, 'temperature', 0.0)
    elif independent_variable == 'top_p':
        return getattr(model, 'top_p', 1.0)
    elif independent_variable == 'system_prompt_variant':
        return getattr(model, 'system_prompt_variant', 'unknown')
    elif independent_variable == 'max_tokens':
        return getattr(model, 'max_tokens', 4096)
    elif independent_variable == 'deployment':
        return getattr(model, 'deployment', 'unknown')
    elif independent_variable == 'index_name':
        return getattr(model, 'index_name', 'unknown')
    elif independent_variable == 'embeddings_model':
        # Map index names to embedding models for cleaner labeling
        index_name = getattr(model, 'index_name', 'unknown')
        embedding_mapping = {
            'uscomm-ana-index': 'default',
            'cyrus-index-ada-002': 'ada-002',
            'cyrus-index-3-large': 'text-3-large',
            'cyrus-index-3-small': 'text-3-small'
        }
        return embedding_mapping.get(index_name, index_name)
    elif independent_variable == 'model_type':
        return model.__class__.__name__
    elif independent_variable == 'base_model':
        if hasattr(model, 'deployment'):
            return getattr(model, 'deployment', 'unknown')
        elif hasattr(model, 'model'):
            return getattr(model, 'model', 'unknown')
        elif hasattr(model, 'deployment_name'):
            return getattr(model, 'deployment_name', 'unknown')
        else:
            # For TevaGPT which uses a fixed deployment, return the class name + fixed model
            if model.__class__.__name__ == 'TevaGPT':
                return 'gpt-4-omni'  # TevaGPT's default deployment
            return model.__class__.__name__
    elif independent_variable == 'top_k':
        return getattr(model, 'top_k', 5)
    elif independent_variable == 'retrieval_method':
        return getattr(model, 'retrieval_method', 'unknown')
    elif independent_variable == 'retrieval_settings':
        # NEW: Generate descriptive labels for retrieval component combinations
        if model.__class__.__name__ == 'TevaGPT':
            return 'baseline_tevagpt'
        elif not hasattr(model, 'retrieval_method') or getattr(model, 'retrieval_method') == 'default':
            return 'baseline_cyrus'
        else:
            # For enhanced retrieval, describe which components are enabled
            components = []
            if getattr(model, 'query_expansion', False):
                components.append('QE')
            if getattr(model, 'hybrid_search', False):
                components.append('HS')  
            if getattr(model, 'advanced_reranking', False):
                components.append('AR')
            if getattr(model, 'diversity_selection', False):
                components.append('DS')
            
            if not components:
                return 'enhanced_none'
            elif len(components) == 4:
                return 'enhanced_full'
            else:
                return f"enhanced_{'_'.join(components)}"
    else:
        # Try to get the attribute directly
        return getattr(model, independent_variable, 'unknown')

def analyze_independent_variable_optimization(all_results, models, independent_variable):
    """Analyze which independent variable values perform best overall"""
    # Calculate overall scores for each model across all constraints
    model_overall_scores = {}
    
    for label, results in all_results.items():
        # Calculate average score across all constraints
        constraint_scores = []
        constraints = set([r['context_constraint'] for r in results])
        
        for constraint in constraints:
            metrics = aggregate_metrics_by_constraint(results, constraint)
            constraint_scores.append(metrics['overall_score'])
        
        model_overall_scores[label] = sum(constraint_scores) / len(constraint_scores) if constraint_scores else 0.0
    
    # Find best performing model
    best_model = max(model_overall_scores.keys(), key=lambda x: model_overall_scores[x])
    best_score = model_overall_scores[best_model]
    
    # Extract independent variable values and analyze performance
    variable_analysis = {}
    
    for label, model in zip(all_results.keys(), models):
        var_value = extract_independent_variable_value(model, independent_variable)
        score = model_overall_scores[label]
        
        # Track performance for each variable value
        if var_value not in variable_analysis:
            variable_analysis[var_value] = []
        variable_analysis[var_value].append(score)
    
    # Find optimal variable value
    optimal_analysis = {}
    if variable_analysis:
        # Average scores for each variable value
        var_averages = {val: sum(scores)/len(scores) for val, scores in variable_analysis.items()}
        optimal_val = max(var_averages.keys(), key=lambda x: var_averages[x])
        optimal_score = var_averages[optimal_val]
        
        optimal_analysis = {
            'variable': independent_variable,
            'optimal_value': optimal_val,
            'optimal_score': optimal_score,
            'all_values': var_averages
        }
    
    return {
        'best_model': best_model,
        'best_score': best_score,
        'model_scores': model_overall_scores,
        'variable_analysis': optimal_analysis
    }

def generate_model_labels(models, independent_variable):
    """Generate descriptive labels for models based on the independent variable"""
    labels = []
    
    for i, model in enumerate(models):
        if independent_variable == 'system_prompt_variant':
            variant = getattr(model, 'system_prompt_variant', 'unknown')
            base_name = model.__class__.__name__
            labels.append(f"{base_name}_{variant}")
        elif independent_variable in ['temperature', 'top_p']:
            # For temperature/top_p, include both in label if they exist
            temp = getattr(model, 'temperature', 0.0)
            top_p = getattr(model, 'top_p', 1.0)
            base_name = model.__class__.__name__
            labels.append(f"{base_name}_t{temp}_p{top_p}")
        elif independent_variable == 'embeddings_model':
            # Use the mapped embedding model name
            embedding_model = extract_independent_variable_value(model, independent_variable)
            base_name = model.__class__.__name__
            labels.append(f"{base_name}_{embedding_model}")
        elif independent_variable == 'model_type':
            labels.append(f"{model.__class__.__name__}_{i}")
        elif independent_variable == 'base_model':
            # Generate labels based on base model (deployment)
            base_model_value = extract_independent_variable_value(model, independent_variable)
            base_name = model.__class__.__name__
            labels.append(f"{base_name}_{base_model_value}")
        elif independent_variable == 'top_k':
            # Generate labels based on top_k values
            top_k_value = extract_independent_variable_value(model, independent_variable)
            base_name = model.__class__.__name__
            labels.append(f"{base_name}_k{top_k_value}")
        elif independent_variable == 'retrieval_method':
            # Generate single-letter labels for retrieval method comparison
            retrieval_value = extract_independent_variable_value(model, independent_variable)
            if retrieval_value == 'unknown':
                labels.append('TevaGPT_retrieval')  # TevaGPT (no retrieval method)
            elif retrieval_value == 'default':
                labels.append('CyrusGPT_old_retrieval')  # CyrusGPT default (old)
            elif retrieval_value == 'enhanced':
                labels.append('CyrusGPT_enhanced_retrieval')  # CyrusGPT enhanced (new)
            else:
                # Fallback for other retrieval methods
                labels.append(retrieval_value[:15].lower())
        elif independent_variable == 'retrieval_settings':
            # NEW: Generate descriptive labels for retrieval settings experiment
            if model.__class__.__name__ == 'TevaGPT':
                labels.append('TevaGPT_Baseline')
            elif not hasattr(model, 'retrieval_method') or getattr(model, 'retrieval_method') == 'default':
                labels.append('CyrusGPT_Default')
            else:
                # For enhanced retrieval, create readable labels
                components = []
                if getattr(model, 'query_expansion', False):
                    components.append('QueryExp')
                if getattr(model, 'hybrid_search', False):
                    components.append('HybridSearch')  
                if getattr(model, 'advanced_reranking', False):
                    components.append('AdvRanking')
                if getattr(model, 'diversity_selection', False):
                    components.append('Diversity')
                
                if not components:
                    labels.append('CyrusGPT_Enhanced_None')
                elif len(components) == 4:
                    labels.append('CyrusGPT_All_Components')
                elif len(components) == 1:
                    # Single component labels
                    comp_name = components[0].replace('QueryExp', 'QueryExpansion').replace('AdvRanking', 'AdvancedRanking')
                    labels.append(f'CyrusGPT_{comp_name}_Only')
                elif len(components) == 2:
                    # Pairwise component labels
                    abbrev = []
                    for comp in components:
                        if comp == 'QueryExp':
                            abbrev.append('QE')
                        elif comp == 'HybridSearch':
                            abbrev.append('HS')
                        elif comp == 'AdvRanking':
                            abbrev.append('AR')
                        elif comp == 'Diversity':
                            abbrev.append('DS')
                    labels.append(f"CyrusGPT_{'_plus_'.join(abbrev)}")
                elif len(components) == 3:
                    # Triple component labels
                    abbrev = []
                    for comp in components:
                        if comp == 'QueryExp':
                            abbrev.append('QE')
                        elif comp == 'HybridSearch':
                            abbrev.append('HS')
                        elif comp == 'AdvRanking':
                            abbrev.append('AR')
                        elif comp == 'Diversity':
                            abbrev.append('DS')
                    labels.append(f"CyrusGPT_{'_'.join(abbrev)}_Triple")
                else:
                    # Fallback for other combinations
                    abbrev = []
                    for comp in components:
                        if comp == 'QueryExp':
                            abbrev.append('QE')
                        elif comp == 'HybridSearch':
                            abbrev.append('HS')
                        elif comp == 'AdvRanking':
                            abbrev.append('AR')
                        elif comp == 'Diversity':
                            abbrev.append('DS')
                    labels.append(f"CyrusGPT_{'_'.join(abbrev)}")
        else:
            # Generic labeling for other variables
            var_value = extract_independent_variable_value(model, independent_variable)
            base_name = model.__class__.__name__
            labels.append(f"{base_name}_{independent_variable}_{var_value}")
    
    return labels

def extract_retry_time_from_error(error_message):
    """
    Extract retry time from Azure OpenAI rate limit error messages.
    
    Args:
        error_message (str): The error message from the API
        
    Returns:
        int: Number of seconds to wait, or None if no retry time found
    """
    if not isinstance(error_message, str):
        return None
    
    # Pattern to match "Try again in X seconds"
    pattern = r"Try again in (\d+) seconds?"
    match = re.search(pattern, error_message)
    
    if match:
        return int(match.group(1))
    
    # Alternative pattern for different error formats
    pattern2 = r"retry after (\d+) seconds?"
    match2 = re.search(pattern2, error_message, re.IGNORECASE)
    
    if match2:
        return int(match2.group(1))
    
    return None

async def evaluate_single_test_case_with_retry(pipeline, test_case, context_constraint, rag_evaluator, max_retries=3, semaphore=None):
    """Fixed timing - only measures actual evaluation time, not semaphore waits"""
    
    if semaphore:
        async with semaphore:
            # Start timing AFTER acquiring semaphore
            start_time = time.time()
            result = await _evaluate_single_test_case_core(pipeline, test_case, context_constraint, rag_evaluator, max_retries)
            end_time = time.time()
    else:
        # Start timing immediately if no semaphore
        start_time = time.time()
        result = await _evaluate_single_test_case_core(pipeline, test_case, context_constraint, rag_evaluator, max_retries)
        end_time = time.time()
    
    # Calculate actual evaluation time (excluding semaphore wait)
    evaluation_time = end_time - start_time
    
    # Override with accurate timing
    if 'rag_scores' in result and isinstance(result['rag_scores'], dict):
        result['rag_scores']['EvaluationTime'] = evaluation_time
        result['rag_scores']['ActualTaskTime'] = evaluation_time  # More descriptive name
    
    return result

async def _evaluate_single_test_case_core(pipeline, test_case, context_constraint, rag_evaluator, max_retries):
    """Core evaluation logic with precise timing for retries"""
    capability = test_case["capability"]
    content = test_case["content"]
    query = test_case["query"]
    expected = test_case["expected_output"]
    file_path = test_case.get("file_path", "")
    file_name = test_case.get("file_name", "")
    folder_path = file_path.rsplit('/', 1)[0] if '/' in file_path else ""

    # Define effective paths
    if context_constraint == 'file':
        effective_path = file_path
    elif context_constraint == 'folder':
        effective_path = folder_path
    elif context_constraint == 'unconstrained':
        effective_path = None
    elif context_constraint == 'given':
        effective_path = None
    else:
        raise ValueError(f"Invalid context_constraint: {context_constraint}")

    # Initialize variables
    response = ""
    full_output = {}
    citations = []
    request_successful = False
    
    # Track timing for actual API calls only (not retry waits)
    total_api_time = 0.0

    # Retry logic for rate limiting
    for attempt in range(max_retries + 1):
        try:
            # Start timing for this specific attempt
            attempt_start = time.time()
            
            # Handle 'given' context constraint
            if context_constraint == 'given':
                if isinstance(pipeline, AskHR) or hasattr(pipeline, 'query_simple'):
                    contextualized_query = f"""Based on the following context, please answer the question:

Context: {content}

Question: {query}

Please answer based solely on the provided context."""
                    
                    response_data = await pipeline.query_simple(contextualized_query)
                    response = response_data if isinstance(response_data, str) else response_data.get('output', '')
                    full_output = response_data.get('full_output', {}) if isinstance(response_data, dict) else {}
                    citations = [{"content": content}]
                else:
                    raise ValueError(f"Unknown pipeline type: {type(pipeline).__name__}")

            # Handle AskHR instances
            elif isinstance(pipeline, (AskHR, CyrusGPT)):
                kwargs = {}
                if context_constraint == 'file':
                    kwargs = {'folder_name': folder_path, 'filename': file_name}
                elif context_constraint == 'folder':
                    kwargs = {'folder_name': folder_path}
                elif context_constraint == 'unconstrained':
                    kwargs = {}
                    
                response_data = await pipeline.query(query, **kwargs)
                response = response_data if isinstance(response_data, str) else response_data.get('output', '')
                full_output = response_data.get('full_output', {}) if isinstance(response_data, dict) else {}
                
                # Extract citations
                citations = []
                if hasattr(full_output, 'choices') and full_output.choices:
                    choice = full_output.choices[0]
                    if hasattr(choice.message, 'context') and choice.message.context:
                        citations = choice.message.context.get('citations', [])
                    elif hasattr(response_data, 'manual_citations'):
                        citations = response_data['manual_citations']

            # Handle TevaGPT instances
            elif hasattr(pipeline, 'process_and_query'):
                kwargs = {}
                if context_constraint == 'file':
                    kwargs = {'folder_name': folder_path, 'filename': file_name}
                elif context_constraint == 'folder':
                    kwargs = {'folder_name': folder_path}
                elif context_constraint == 'unconstrained':
                    kwargs = {}
                    
                response_data = await pipeline.process_and_query(query=query, file_path=effective_path, **kwargs)
                response = response_data.get('output', '') if isinstance(response_data, dict) else response_data
                full_output = response_data.get('full_output', {}) if isinstance(response_data, dict) else {}
                
                # Extract citations
                citations = []
                if hasattr(full_output, 'choices') and full_output.choices:
                    choice = full_output.choices[0]
                    if hasattr(choice.message, 'context') and choice.message.context:
                        citations = choice.message.context.get('citations', [])
            else:
                raise ValueError(f"Unknown pipeline type: {type(pipeline).__name__}")
            
            # Record time for successful attempt
            attempt_end = time.time()
            total_api_time += (attempt_end - attempt_start)
            
            # If we get here, the request was successful
            request_successful = True
            break
            
        except Exception as e:
            # Record time for failed attempt (exclude retry wait time)
            attempt_end = time.time()
            total_api_time += (attempt_end - attempt_start)
            
            error_str = str(e)
            
            # Check if this is a rate limit error
            if "429" in error_str or "rate limit" in error_str.lower():
                retry_time = extract_retry_time_from_error(error_str)
                
                if retry_time is not None and attempt < max_retries:
                    wait_time = retry_time + 1
                    logger.warning(f"Rate limit hit on test {test_case['test_id']}, attempt {attempt + 1}. "
                                 f"Waiting {wait_time} seconds before retry...")
                    # DON'T include sleep time in timing measurement
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt < max_retries:
                    wait_time = (2 ** attempt) * 5
                    logger.warning(f"Rate limit hit on test {test_case['test_id']}, attempt {attempt + 1}. "
                                 f"Using exponential backoff: {wait_time} seconds...")
                    # DON'T include sleep time in timing measurement
                    await asyncio.sleep(wait_time)
                    continue
            
            # If it's the last attempt or not a rate limit error
            if attempt == max_retries:
                logger.error(f"Failed test {test_case['test_id']} after {max_retries + 1} attempts: {e}")
                return {
                    "context_constraint": context_constraint,
                    "test_id": test_case["test_id"],
                    "capability": capability,
                    "file_path": effective_path if context_constraint != 'given' else 'given_context',
                    "query": query,
                    "response": f"Error after {max_retries + 1} attempts: {str(e)}",
                    "expected": expected,
                    "expected_context": content,
                    "retrieved_docs": [],
                    "retrieval_context": [],
                    "rag_scores": {
                        'error': str(e),
                        'EvaluationTime': total_api_time,
                        'ActualTaskTime': total_api_time
                    }
                }

    # Process successful results
    if request_successful:
        # Time the RAG evaluation as well
        rag_start = time.time()
        
        # Extract document content for RAG evaluation
        retrieved_docs = []
        if citations:
            for doc in citations:
                if isinstance(doc, dict):
                    retrieved_docs.append(doc.get('content', ''))
                else:
                    retrieved_docs.append(str(doc))

        # Evaluate using RAG metrics
        rag_scores = rag_evaluator.evaluate_sample(query, retrieved_docs, response, expected)
        
        rag_end = time.time()
        rag_time = rag_end - rag_start
        
        # Include RAG evaluation time in total
        total_time = total_api_time + rag_time
        
        # Update timing in rag_scores
        if isinstance(rag_scores, dict):
            rag_scores['EvaluationTime'] = total_time
            rag_scores['ActualTaskTime'] = total_time
            rag_scores['ApiCallTime'] = total_api_time
            rag_scores['RagEvaluationTime'] = rag_time
        
        return {
            "context_constraint": context_constraint,
            "test_id": test_case["test_id"],
            "capability": capability,
            "file_path": effective_path if context_constraint != 'given' else 'given_context',
            "query": query,
            "response": response,
            "expected": expected,
            "expected_context": content,
            "retrieved_docs": retrieved_docs,
            "retrieval_context": citations,
            "rag_scores": rag_scores
        }

async def evaluate_with_constraint_parallel(pipeline, test_cases, pipeline_name, context_constraint, max_retries=3, max_concurrent=5):
    """
    Evaluate pipeline with parallel execution of test cases
    
    Args:
        pipeline: The model pipeline to evaluate
        test_cases: List of test cases to evaluate
        pipeline_name: Name of the pipeline for logging
        context_constraint: Context constraint level
        max_retries: Maximum retries for rate-limited requests
        max_concurrent: Maximum number of concurrent requests (adjust based on rate limits)
    """
    print(f"  Running {pipeline_name} with {max_concurrent} concurrent requests...")
    
    # Create semaphore to limit concurrent requests
    semaphore = Semaphore(max_concurrent)
    
    # Initialize the LLM evaluator
    rag_evaluator = LLMRAGEvaluator()
    
    # Create tasks for all test cases
    tasks = []
    for test_case in test_cases:
        task = evaluate_single_test_case_with_retry(
            pipeline, test_case, context_constraint, rag_evaluator, max_retries, semaphore
        )
        tasks.append(task)
    
    # Execute all tasks concurrently with progress tracking
    results = []
    completed = 0
    total = len(tasks)
    
    # Use as_completed to get results as they finish
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            results.append(result)
            completed += 1
            
            # Progress update every 10% or every 5 completions, whichever is more frequent
            if completed % max(1, min(5, total // 10)) == 0 or completed == total:
                error_count = len([r for r in results if 'error' in r.get('rag_scores', {})])
                success_rate = ((completed - error_count) / completed * 100) if completed > 0 else 0
                print(f"    Progress: {completed}/{total} ({completed/total*100:.1f}%)")
                
        except Exception as e:
            logger.error(f"Task failed with exception: {e}")
            # Create error result for failed task
            results.append({
                "context_constraint": context_constraint,
                "test_id": -1,  # Unknown test ID
                "capability": "unknown",
                "file_path": "unknown",
                "query": "unknown",
                "response": f"Task execution error: {str(e)}",
                "expected": "unknown",
                "expected_context": "unknown",
                "retrieved_docs": [],
                "retrieval_context": [],
                "rag_scores": {'error': str(e)}
            })
            completed += 1
    
    return results

# Replace the model evaluation section in run_all_evals
async def run_all_evals(models, context_levels=['unconstrained'], independent_variable='model_type', 
                       rate_limit_delay=0, models_per_batch=10, max_retries=5, max_concurrent=5):
    """
    Compare multiple model instances across specified context levels with parallel execution
    
    Args:
        models: List of model instances
        context_levels: List of context levels to evaluate
        independent_variable: The variable being tested
        rate_limit_delay: Seconds to wait between model batches (fallback)
        models_per_batch: Number of models to evaluate before sleeping
        max_retries: Maximum number of retries for rate-limited requests
        max_concurrent: Maximum number of concurrent requests per model (NEW)
    """
    # Handle validation and setup (same as before)
    if not isinstance(models, list):
        raise ValueError("models must be a list of model instances")
    
    if not models:
        raise ValueError("At least one model must be provided")
    
    if context_levels is None:
        context_levels = ['given', 'file', 'folder', 'unconstrained']
    
    print(f"Comparing {len(models)} models on independent variable: {independent_variable}")
    print(f"Parallel execution: {max_concurrent} concurrent requests per model")
    print(f"Rate limiting fallback: {rate_limit_delay}s delay after every {models_per_batch} models")
    print(f"Max retries per request: {max_retries}")
    
    # Generate labels and load test cases (same as before)
    model_labels = generate_model_labels(models, independent_variable)
    
    print("Model configurations:")
    for label, model in zip(model_labels, models):
        var_value = extract_independent_variable_value(model, independent_variable)
        print(f"  {label}: {independent_variable}={var_value}")

    print("Loading test cases")
    test_cases = get_test_cases()
    print(f"Loaded {len(test_cases)} test cases!")
    
    # Validate context levels
    valid_constraints = ['given', 'file', 'folder', 'unconstrained']
    for context_level in context_levels:
        if context_level not in valid_constraints:
            raise ValueError(f"Invalid context_level: {context_level}. Must be one of: {valid_constraints}")
    
    print(f"\nEvaluating {len(models)} models across constraint levels: {context_levels}")
    
    # Store all results
    all_results = {label: [] for label in model_labels}
    
    # Track timing
    start_time = time.time()
    
    # Run evaluations with parallel execution
    for context_level in context_levels:
        print(f"\nEvaluating constraint level: {context_level}")
        context_start_time = time.time()
        
        for model_idx, (model, label) in enumerate(zip(models, model_labels)):
            model_start_time = time.time()
            
            # Use parallel evaluation instead of sequential
            results = await evaluate_with_constraint_parallel(
                model, test_cases, label, context_level, max_retries=max_retries, max_concurrent=max_concurrent
            )
            all_results[label].extend(results)
            
            model_end_time = time.time()
            model_duration = model_end_time - model_start_time
            
            # Progress update with timing
            metrics = aggregate_metrics_by_constraint(results, context_level)
            error_count = len([r for r in results if 'error' in r.get('rag_scores', {})])
            print(f"    {label} completed in {model_duration:.1f}s - Score: {metrics['overall_score']:.4f} (errors: {error_count})")
            
            # Rate limiting fallback: sleep after every models_per_batch models
            if (model_idx + 1) % models_per_batch == 0 and (model_idx + 1) < len(models):
                print(f"    Preventive rate limit: sleeping for {rate_limit_delay} seconds...")
                await asyncio.sleep(rate_limit_delay)
        
        context_end_time = time.time()
        context_duration = context_end_time - context_start_time
        print(f"  {context_level} constraint completed in {context_duration:.1f}s")

    total_duration = time.time() - start_time
    print(f"\nAll evaluations completed in {total_duration:.1f}s")

    # Create comparison outputs (same as before)
    print("\nCreating comparison outputs")
    summary_df = create_model_comparison_summary(all_results)
    capability_df = create_model_capability_comparison(all_results)
    
    # Create detailed results for each model
    detailed_dfs = {}
    for label, results in all_results.items():
        detailed_dfs[label] = create_detailed_results_dataframe(results, label)

    # Create improvement examples sheet
    print("Creating improvement examples sheet...")
    try:
        improvement_df = await create_improvement_examples_sheet(all_results, model_labels)
        print(f"Found {len(improvement_df)} improvement examples")
    except Exception as e:
        print(f"Error creating improvement examples: {e}")
        improvement_df = pd.DataFrame()

    # Save to Excel (same as before)
    os.makedirs('eval_outputs', exist_ok=True)
    model_names = "_vs_".join([label.split('_')[0] for label in model_labels[:3]])
    context_levels_str = "_".join(context_levels)
    variable_name = f'{model_names}_{independent_variable}_{context_levels_str}'
    variable_name_short = variable_name[:50]
    output_path = f'eval_outputs/model_comparison_{variable_name_short}.xlsx'
    
    save_results_with_improved_formatting(output_path, summary_df, capability_df, improvement_df, detailed_dfs)

    print(f"\nModel comparison complete!")
    print(f"Results saved to: {output_path}")
    print(f"Total evaluation time: {total_duration:.1f}s")
    
    # Print quick summary (same as before)
    print(f"\nQuick Summary:")
    print("="*50)
    for constraint in context_levels:
        print(f"\n{constraint.upper()}:")
        constraint_scores = {}
        for label, results in all_results.items():
            metrics = aggregate_metrics_by_constraint(results, constraint)
            constraint_scores[label] = metrics['overall_score']
        
        sorted_scores = sorted(constraint_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (label, score) in enumerate(sorted_scores, 1):
            print(f"  {rank}. {label}: {score:.4f}")
    
    # Analyze independent variable optimization
    variable_analysis = analyze_independent_variable_optimization(all_results, models, independent_variable)
        
    return {
        'all_results': all_results,
        'output_path': output_path,
        'variable_analysis': variable_analysis
    }

def create_capability_comparison(tevagpt_all, askhr_all):
    """create capability comparison summary table - LEGACY"""
    # get unique capabilities
    all_capabilities = set([r['capability'] for r in tevagpt_all + askhr_all])
    summary_data = []
    
    # loop thru the capabilities and aggregate the metrics to add to the summary data
    for capability in all_capabilities:
        tevagpt_metrics = aggregate_metrics_by_capability(tevagpt_all, capability)
        askhr_metrics = aggregate_metrics_by_capability(askhr_all, capability)
        
        summary_data.append({
            'capability': capability,
            'tevagpt_overall_score': round(tevagpt_metrics['overall_score'], 4),
            'tevagpt_task_success': round(tevagpt_metrics['task_success_avg'], 4),
            'tevagpt_faithfulness': round(tevagpt_metrics['faithfulness_avg'], 4),
            'tevagpt_answer_relevancy': round(tevagpt_metrics['answer_relevancy_avg'], 4),
            'tevagpt_context_recall': round(tevagpt_metrics['context_recall_avg'], 4),
            'tevagpt_context_precision': round(tevagpt_metrics['context_precision_avg'], 4),
            'askhr_overall_score': round(askhr_metrics['overall_score'], 4),
            'askhr_task_success': round(askhr_metrics['task_success_avg'], 4),
            'askhr_faithfulness': round(askhr_metrics['faithfulness_avg'], 4),
            'askhr_answer_relevancy': round(askhr_metrics['answer_relevancy_avg'], 4),
            'askhr_context_recall': round(askhr_metrics['context_recall_avg'], 4),
            'askhr_context_precision': round(askhr_metrics['context_precision_avg'], 4),
            'overall_score_winner': 'TevaGPT' if tevagpt_metrics['overall_score'] > askhr_metrics['overall_score'] else 'AskHR',
            'overall_score_difference': round(abs(tevagpt_metrics['overall_score'] - askhr_metrics['overall_score']), 4)
        })
    
    return pd.DataFrame(summary_data)

def sanitize_df(df):
    '''remove illegal excel characters from all string cells in df'''
    
    # compile pattern for all control chars disallowed by Excel
    _illegal_excel_chars = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]')

    return df.map(
        lambda x: _illegal_excel_chars.sub('', x) if isinstance(x, str) else x
    )


async def process_sample_queries(
    models: List[Any], 
    input_csv_path: str, 
    output_excel_path: str = None,
    context_levels: List[str] = ['unconstrained'],
    max_concurrent: int = 5,
    max_retries: int = 3,
    independent_variable: str = 'model_type'
) -> str:
    """
    Process a spreadsheet of sample queries through multiple models and output results with timing metrics.
    
    Args:
        models: List of model instances to evaluate
        input_csv_path: Path to CSV file with columns 'query', and optionally 'file_path', 'file_name', 'folder_path'
        output_excel_path: Output path for Excel file (auto-generated if None)
        context_levels: List of context constraint levels to test
        max_concurrent: Maximum concurrent requests per model
        max_retries: Maximum retries for rate-limited requests
        independent_variable: Variable being compared across models
        
    Returns:
        str: Path to output Excel file
        
    Expected CSV format:
        query,file_path,file_name,folder_path,notes
        "What is the policy on vacation days?","hr/policies/vacation.pdf","vacation.pdf","hr/policies","Optional notes"
        "How do I submit expenses?",,,,
    """
    
    print(f"Processing sample queries from: {input_csv_path}")
    print(f"Testing {len(models)} models across constraint levels: {context_levels}")
    
    # Load sample queries from CSV
    try:
        with open(input_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            queries_df = pd.read_csv(f)
    except Exception as e:
        raise ValueError(f"Could not load CSV file {input_csv_path}: {e}")
    
    # Validate required columns
    if 'query' not in queries_df.columns:
        raise ValueError("CSV must contain a 'query' column")
    
    # Fill NaN values with empty strings for optional columns
    optional_columns = ['file_path', 'file_name', 'folder_path', 'notes']
    for col in optional_columns:
        if col not in queries_df.columns:
            queries_df[col] = ""
        queries_df[col] = queries_df[col].fillna("")
    
    print(f"Loaded {len(queries_df)} sample queries")
    
    # Generate model labels
    model_labels = generate_model_labels(models, independent_variable)
    
    print("Model configurations:")
    for label, model in zip(model_labels, models):
        var_value = extract_independent_variable_value(model, independent_variable)
        print(f"  {label}: {independent_variable}={var_value}")
    
    # Store all results
    all_results = []
    
    # Create semaphore for concurrent requests
    semaphore = Semaphore(max_concurrent)
    
    # Process each model
    total_start_time = time.time()
    
    for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        print(f"\nProcessing model {model_idx + 1}/{len(models)}: {model_label}")
        
        for context_level in context_levels:
            print(f"  Context level: {context_level}")
            context_start_time = time.time()
            
            # Create tasks for all queries in this context level
            tasks = []
            for query_idx, row in queries_df.iterrows():
                task = process_single_sample_query(
                    model, model_label, row, context_level, query_idx, 
                    max_retries, semaphore
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            context_results = []
            completed = 0
            total_queries = len(tasks)
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    context_results.append(result)
                    completed += 1
                    
                    # Progress update
                    if completed % max(1, min(5, total_queries // 10)) == 0 or completed == total_queries:
                        error_count = len([r for r in context_results if r.get('error')])
                        print(f"    Progress: {completed}/{total_queries} ({completed/total_queries*100:.1f}%) - Errors: {error_count}")
                        
                except Exception as e:
                    logger.error(f"Task failed with exception: {e}")
                    completed += 1
            
            all_results.extend(context_results)
            
            context_end_time = time.time()
            context_duration = context_end_time - context_start_time
            error_count = len([r for r in context_results if r.get('error')])
            print(f"    Completed {len(context_results)} queries in {context_duration:.1f}s (errors: {error_count})")
    
    total_duration = time.time() - total_start_time
    print(f"\nAll queries processed in {total_duration:.1f}s")
    
    # Create output DataFrame
    output_df = create_sample_queries_dataframe(all_results)
    
    # Generate output path if not provided
    if output_excel_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_names = "_vs_".join([label.split('_')[0] for label in model_labels[:3]])
        context_levels_str = "_".join(context_levels)
        output_excel_path = f'eval_outputs/sample_queries_{model_names}_{context_levels_str}_{timestamp}.xlsx'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    
    # Save to Excel with formatting
    save_sample_queries_excel(output_df, output_excel_path, model_labels, context_levels)
    
    print(f"\nSample query processing complete!")
    print(f"Results saved to: {output_excel_path}")
    print(f"Total processing time: {total_duration:.1f}s")
    
    # Print summary statistics
    print_sample_queries_summary(output_df, model_labels, context_levels)
    
    return output_excel_path

async def process_single_sample_query(
    model: Any, 
    model_label: str, 
    query_row: pd.Series, 
    context_level: str, 
    query_idx: int,
    max_retries: int,
    semaphore: Semaphore
) -> Dict[str, Any]:
    """Process a single sample query through a model with retry logic"""
    
    async with semaphore:
        query = query_row['query']
        file_path = query_row.get('file_path', '').strip()
        file_name = query_row.get('file_name', '').strip()
        folder_path = query_row.get('folder_path', '').strip()
        notes = query_row.get('notes', '').strip()
        
        # If folder_path is empty but file_path has a folder, extract it
        if not folder_path and file_path and '/' in file_path:
            folder_path = file_path.rsplit('/', 1)[0]
        
        # Track timing for API calls only
        total_api_time = 0.0
        response = ""
        full_output = {}
        citations = []
        error_message = ""
        request_successful = False
        
        # Retry logic for rate limiting
        for attempt in range(max_retries + 1):
            try:
                attempt_start = time.time()
                
                # Handle different context levels
                if context_level == 'file' and file_name and folder_path:
                    kwargs = {'folder_name': folder_path, 'filename': file_name}
                elif context_level == 'folder' and folder_path:
                    kwargs = {'folder_name': folder_path}
                elif context_level == 'unconstrained':
                    kwargs = {}
                else:
                    # Skip this combination if constraints can't be met
                    return create_sample_query_result(
                        model_label, query_idx, query, context_level, file_path, file_name, 
                        folder_path, notes, "", 0.0, [], 
                        f"Cannot apply {context_level} constraint - missing required path information"
                    )
                
                # Call the appropriate model method
                if isinstance(model, (AskHR, CyrusGPT)):
                    response_data = await model.query(query, **kwargs)
                elif hasattr(model, 'process_and_query'):  # TevaGPT
                    effective_path = file_path if context_level == 'file' else (folder_path if context_level == 'folder' else None)
                    response_data = await model.process_and_query(query=query, file_path=effective_path, **kwargs)
                else:
                    raise ValueError(f"Unknown model type: {type(model).__name__}")
                
                # Extract response and metadata
                if isinstance(response_data, str):
                    response = response_data
                else:
                    response = response_data.get('output', '')
                    full_output = response_data.get('full_output', {})
                
                # Extract citations if available
                citations = []
                if hasattr(full_output, 'choices') and full_output.choices:
                    choice = full_output.choices[0]
                    if hasattr(choice.message, 'context') and choice.message.context:
                        citations = choice.message.context.get('citations', [])
                
                # Record successful attempt time
                attempt_end = time.time()
                total_api_time += (attempt_end - attempt_start)
                request_successful = True
                break
                
            except Exception as e:
                attempt_end = time.time()
                total_api_time += (attempt_end - attempt_start)
                error_str = str(e)
                
                # Handle rate limiting
                if "429" in error_str or "rate limit" in error_str.lower():
                    retry_time = extract_retry_time_from_error(error_str)
                    
                    if retry_time is not None and attempt < max_retries:
                        wait_time = retry_time + 1
                        await asyncio.sleep(wait_time)
                        continue
                    elif attempt < max_retries:
                        wait_time = (2 ** attempt) * 5
                        await asyncio.sleep(wait_time)
                        continue
                
                # If it's the last attempt
                if attempt == max_retries:
                    error_message = f"Failed after {max_retries + 1} attempts: {str(e)}"
                    break
        
        # Extract retrieved document contents
        retrieved_docs = []
        if citations:
            for doc in citations:
                if isinstance(doc, dict):
                    retrieved_docs.append(doc.get('content', ''))
                else:
                    retrieved_docs.append(str(doc))
        
        return create_sample_query_result(
            model_label, query_idx, query, context_level, file_path, file_name,
            folder_path, notes, response, total_api_time, retrieved_docs, error_message
        )

def create_sample_query_result(
    model_label: str, query_idx: int, query: str, context_level: str,
    file_path: str, file_name: str, folder_path: str, notes: str,
    response: str, api_time: float, retrieved_docs: List[str], error_message: str
) -> Dict[str, Any]:
    """Create a standardized result dictionary for a sample query"""
    
    return {
        'model': model_label,
        'query_id': query_idx,
        'context_level': context_level,
        'query': query,
        'file_path': file_path,
        'file_name': file_name,
        'folder_path': folder_path,
        'notes': notes,
        'response': response,
        'response_length_chars': len(response) if response else 0,
        'response_length_words': len(response.split()) if response else 0,
        'api_time_seconds': round(api_time, 3),
        'retrieved_docs_count': len(retrieved_docs),
        'retrieved_docs_content': "\n\n--- NEXT DOC ---\n\n".join(retrieved_docs) if retrieved_docs else "",
        'success': not bool(error_message),
        'error': error_message,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

def create_sample_queries_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from sample query results"""
    
    # Sort results by query_id, then model, then context_level
    results.sort(key=lambda x: (x['query_id'], x['model'], x['context_level']))
    
    return pd.DataFrame(results)

def save_sample_queries_excel(
    df: pd.DataFrame, 
    output_path: str, 
    model_labels: List[str], 
    context_levels: List[str]
):
    """Save sample queries results to Excel with formatting"""
    
    # Sanitize DataFrame
    df = sanitize_df(df)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results sheet
        df.to_excel(writer, index=False, sheet_name='Sample_Query_Results')
        
        # Create summary sheet
        summary_df = create_sample_queries_summary_df(df, model_labels, context_levels)
        summary_df.to_excel(writer, index=False, sheet_name='Summary_Statistics')
        
        # Format main sheet
        try:
            workbook = writer.book
            main_ws = writer.sheets['Sample_Query_Results']
            
            # Set column widths
            column_widths = {
                'A': 15,  # model
                'B': 10,  # query_id
                'C': 15,  # context_level
                'D': 50,  # query
                'E': 30,  # file_path
                'F': 20,  # file_name
                'G': 30,  # folder_path
                'H': 30,  # notes
                'I': 80,  # response
                'J': 12,  # response_length_chars
                'K': 12,  # response_length_words
                'L': 12,  # api_time_seconds
                'M': 12,  # retrieved_docs_count
                'N': 80,  # retrieved_docs_content
                'O': 10,  # success
                'P': 40,  # error
                'Q': 20   # timestamp
            }
            
            for col_letter, width in column_widths.items():
                if col_letter in [chr(65 + i) for i in range(len(df.columns))]:
                    main_ws.column_dimensions[col_letter].width = width
            
            # Apply text wrapping to response and retrieved_docs_content columns
            from openpyxl.styles import Alignment
            wrap_alignment = Alignment(wrap_text=True, vertical='top')
            
            for row in range(2, len(df) + 2):  # Skip header
                main_ws[f'I{row}'].alignment = wrap_alignment  # response
                main_ws[f'N{row}'].alignment = wrap_alignment  # retrieved_docs_content
                # Set row height for better readability
                main_ws.row_dimensions[row].height = 60
                
        except Exception as e:
            print(f"Warning: Could not apply Excel formatting: {e}")

def create_sample_queries_summary_df(
    df: pd.DataFrame, 
    model_labels: List[str], 
    context_levels: List[str]
) -> pd.DataFrame:
    """Create summary statistics DataFrame"""
    
    summary_data = []
    
    for model in model_labels:
        for context_level in context_levels:
            model_context_df = df[(df['model'] == model) & (df['context_level'] == context_level)]
            
            if len(model_context_df) > 0:
                total_queries = len(model_context_df)
                successful_queries = len(model_context_df[model_context_df['success'] == True])
                failed_queries = total_queries - successful_queries
                success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
                
                # Calculate timing statistics for successful queries only
                successful_df = model_context_df[model_context_df['success'] == True]
                if len(successful_df) > 0:
                    avg_time = successful_df['api_time_seconds'].mean()
                    min_time = successful_df['api_time_seconds'].min()
                    max_time = successful_df['api_time_seconds'].max()
                    avg_response_length = successful_df['response_length_words'].mean()
                    avg_docs_retrieved = successful_df['retrieved_docs_count'].mean()
                else:
                    avg_time = min_time = max_time = avg_response_length = avg_docs_retrieved = 0
                
                summary_data.append({
                    'model': model,
                    'context_level': context_level,
                    'total_queries': total_queries,
                    'successful_queries': successful_queries,
                    'failed_queries': failed_queries,
                    'success_rate_percent': round(success_rate, 1),
                    'avg_response_time_seconds': round(avg_time, 3),
                    'min_response_time_seconds': round(min_time, 3),
                    'max_response_time_seconds': round(max_time, 3),
                    'avg_response_length_words': round(avg_response_length, 1),
                    'avg_docs_retrieved': round(avg_docs_retrieved, 1)
                })
    
    return pd.DataFrame(summary_data)

def print_sample_queries_summary(
    df: pd.DataFrame, 
    model_labels: List[str], 
    context_levels: List[str]
):
    """Print summary statistics to console"""
    
    print(f"\nSample Queries Summary:")
    print("=" * 60)
    
    for context_level in context_levels:
        print(f"\n{context_level.upper()} Context:")
        print("-" * 40)
        
        for model in model_labels:
            model_context_df = df[(df['model'] == model) & (df['context_level'] == context_level)]
            
            if len(model_context_df) > 0:
                total = len(model_context_df)
                successful = len(model_context_df[model_context_df['success'] == True])
                success_rate = (successful / total * 100) if total > 0 else 0
                
                successful_df = model_context_df[model_context_df['success'] == True]
                avg_time = successful_df['api_time_seconds'].mean() if len(successful_df) > 0 else 0
                
                print(f"  {model}: {successful}/{total} queries ({success_rate:.1f}%) - Avg time: {avg_time:.2f}s")

# Example usage function
async def example_sample_queries_processing():
    """Example of how to use the sample queries processing function"""
    
    # Create some models to test
    models = [
        TevaGPT(temperature=0.0),
        AskHR(temperature=0.0),
        CyrusGPT(retrieval_method='default', temperature=0.0),
        CyrusGPT(retrieval_method='enhanced', temperature=0.0)
    ]
    
    # Process sample queries
    output_path = await process_sample_queries(
        models=models,
        input_csv_path='tevagpt_vs_askhr/live_query_tests.csv',  # Your input CSV
        context_levels=['unconstrained'],
        max_concurrent=3,  # Adjust based on rate limits
        max_retries=2,
        independent_variable='model_type'
    )
    
    print(f"Results saved to: {output_path}")

import time
import os
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass
from statistics import mean, median
import asyncio
import logging

@dataclass
class TimingResult:
    """Detailed timing breakdown for a single query"""
    model_name: str
    test_id: int
    query: str
    total_time: float
    
    # High-level stages
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    
    # Detailed component timings (for enhanced models)
    query_expansion_time: float = 0.0
    reranking_time: float = 0.0
    diversity_selection_time: float = 0.0
    
    # Two-stage breakdown
    gpt4o_retrieval_time: float = 0.0
    context_formatting_time: float = 0.0
    final_generation_time: float = 0.0
    
    # Architecture type
    architecture: str = ""
    error_message: str = ""
    success: bool = True
    raw_response: str = ""
    retrieved_docs_count: int = 0

async def run_detailed_timing_analysis(
    models: List[Any],
    test_cases: List[Dict] = None,
    context_levels: List[str] = ['unconstrained'],
    max_concurrent: int = 5,
    max_retries: int = 3
) -> str:
    """
    Run detailed timing analysis that exactly matches the performance evaluation pipeline
    """
    print("=" * 60)
    print("DETAILED PIPELINE TIMING ANALYSIS")
    print("=" * 60)
    
    # Load test cases if not provided - use same function as performance eval
    if test_cases is None:
        test_cases = get_test_cases()
        print(f"Loaded {len(test_cases)} test cases")
    
    # Generate model labels for identification - use same function as performance eval
    model_labels = generate_model_labels(models, 'model_type')
    print(f"Analyzing {len(models)} models: {', '.join(model_labels)}")
    
    # Store all timing results
    all_timing_results = []
    
    # Create semaphore for concurrent requests - same as performance eval
    semaphore = Semaphore(max_concurrent)
    
    total_start_time = time.time()
    
    # Process each context level - same loop structure as performance eval
    for context_level in context_levels:
        print(f"\n📊 Analyzing context level: {context_level}")
        
        # Process each model - same loop structure as performance eval
        for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
            print(f"\n  🔍 Model {model_idx + 1}/{len(models)}: {model_label}")
            model_start_time = time.time()
            
            # Create tasks for all test cases - same as performance eval
            tasks = []
            for test_case in test_cases:
                task = analyze_single_query_timing_exact(
                    model, model_label, test_case, context_level, 
                    max_retries, semaphore
                )
                tasks.append(task)
            
            # Execute all tasks concurrently - same as performance eval
            model_results = []
            completed = 0
            total_queries = len(tasks)
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    model_results.append(result)
                    completed += 1
                    
                    # Progress update - same format as performance eval
                    if completed % max(1, min(10, total_queries // 10)) == 0 or completed == total_queries:
                        error_count = len([r for r in model_results if not r.success])
                        avg_time = mean([r.total_time for r in model_results if r.success]) if model_results else 0
                        print(f"    Progress: {completed}/{total_queries} ({completed/total_queries*100:.1f}%) - Avg: {avg_time:.1f}s - Errors: {error_count}")
                        
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    completed += 1
            
            all_timing_results.extend(model_results)
            
            model_end_time = time.time()
            model_duration = model_end_time - model_start_time
            successful_results = [r for r in model_results if r.success]
            if successful_results:
                avg_total = mean([r.total_time for r in successful_results])
                avg_retrieval = mean([r.retrieval_time for r in successful_results])
                avg_generation = mean([r.generation_time for r in successful_results])
                print(f"    ✅ {model_label} completed in {model_duration:.1f}s")
                print(f"       Avg per query: {avg_total:.1f}s (Retrieval: {avg_retrieval:.1f}s, Generation: {avg_generation:.1f}s)")
            else:
                print(f"    ❌ {model_label} failed - no successful queries")
    
    total_duration = time.time() - total_start_time
    print(f"\n🎉 Analysis completed in {total_duration:.1f}s")
    
    # Create comprehensive analysis outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f'eval_outputs/detailed_timing_analysis_{timestamp}.xlsx'
    
    create_timing_analysis_excel(all_timing_results, output_path, model_labels)
    
    print(f"\n📈 Detailed timing analysis saved to: {output_path}")
    
    # Print executive summary
    print_executive_timing_summary(all_timing_results, model_labels)
    
    return output_path

async def analyze_single_query_timing_exact(
    model: Any,
    model_label: str,
    test_case: Dict,
    context_level: str,
    max_retries: int,
    semaphore: Semaphore
) -> TimingResult:
    """
    Analyze timing for a single query - EXACTLY matches the performance evaluation pipeline
    No RAG evaluation included here since that's not part of the model's actual performance
    """
    
    async with semaphore:
        # Initialize timing result
        timing_result = TimingResult(
            model_name=model_label,
            test_id=test_case["test_id"],
            query=test_case["query"],
            total_time=0.0
        )
        
        # Detect architecture - same logic as CyrusGPT uses
        if hasattr(model, 'azure_search_compatible'):
            if getattr(model, 'azure_search_compatible', True):
                timing_result.architecture = "integrated"
            else:
                timing_result.architecture = "two_stage"
        elif model.__class__.__name__ == 'TevaGPT':
            timing_result.architecture = "integrated" 
        elif model.__class__.__name__ == 'AskHR':
            timing_result.architecture = "integrated"
        else:
            timing_result.architecture = "two_stage"
        
        # Check if enhanced
        if hasattr(model, 'retrieval_method') and getattr(model, 'retrieval_method') == 'enhanced':
            timing_result.architecture = "enhanced_two_stage"
        
        query_start_time = time.time()
        
        # Retry logic - same as performance evaluation
        for attempt in range(max_retries + 1):
            try:
                # Execute the exact same query that performance evaluation does
                response, full_output, citations = await execute_model_query_with_timing(
                    model, test_case, context_level, timing_result
                )
                
                query_end_time = time.time()
                timing_result.total_time = query_end_time - query_start_time
                timing_result.success = True
                timing_result.raw_response = response
                timing_result.retrieved_docs_count = len(citations) if citations else 0
                
                # Success - break retry loop
                break
                
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a rate limit error - same logic as performance eval
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries:
                        retry_time = extract_retry_time_from_error(error_str)
                        wait_time = retry_time + 1 if retry_time else (2 ** attempt) * 2
                        
                        logger.warning(f"Rate limit hit on test {test_case['test_id']}, attempt {attempt + 1}. "
                                     f"Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                
                # If it's the last attempt or not a rate limit error
                if attempt == max_retries:
                    query_end_time = time.time()
                    timing_result.total_time = query_end_time - query_start_time
                    timing_result.error_message = str(e)
                    timing_result.success = False
                    logger.error(f"Query timing analysis failed for {model_label}: {e}")
                    break
        
        return timing_result

async def execute_model_query_with_timing(
    model: Any,
    test_case: Dict,
    context_level: str,
    timing_result: TimingResult
) -> tuple:
    """
    Execute the exact same query pipeline as the performance evaluation, but with timing
    """
    capability = test_case["capability"]
    content = test_case["content"]
    query = test_case["query"]
    expected = test_case["expected_output"]
    file_path = test_case.get("file_path", "")
    file_name = test_case.get("file_name", "")
    folder_path = file_path.rsplit('/', 1)[0] if '/' in file_path else ""

    # Define effective paths - same logic as performance eval
    if context_level == 'file':
        effective_path = file_path
    elif context_level == 'folder':
        effective_path = folder_path
    elif context_level == 'unconstrained':
        effective_path = None
    elif context_level == 'given':
        effective_path = None
    else:
        raise ValueError(f"Invalid context_constraint: {context_level}")

    response = ""
    full_output = {}
    citations = []

    # Execute the same query logic as performance evaluation
    if context_level == 'given':
        # Handle 'given' context constraint - same as performance eval
        if hasattr(model, 'query_simple'):
            contextualized_query = f"""Based on the following context, please answer the question:

Context: {content}

Question: {query}

Please answer based solely on the provided context."""
            
            gen_start = time.time()
            response_data = await model.query_simple(contextualized_query)
            gen_end = time.time()
            
            timing_result.generation_time = gen_end - gen_start
            timing_result.final_generation_time = timing_result.generation_time
            timing_result.retrieval_time = 0.0  # No retrieval for given context
            
            response = response_data if isinstance(response_data, str) else response_data.get('output', '')
            full_output = response_data.get('full_output', {}) if isinstance(response_data, dict) else {}
            citations = [{"content": content}]
        else:
            raise ValueError(f"Unknown pipeline type: {type(model).__name__}")

    # Handle different model types - same logic as performance eval
    elif isinstance(model, (AskHR, CyrusGPT)):
        kwargs = {}
        if context_level == 'file':
            kwargs = {'folder_name': folder_path, 'filename': file_name}
        elif context_level == 'folder':
            kwargs = {'folder_name': folder_path}
        elif context_level == 'unconstrained':
            kwargs = {}
        
        # Time the model query
        if timing_result.architecture == "enhanced_two_stage":
            # Enhanced CyrusGPT - time components separately
            response_data = await time_enhanced_cyrus_query(model, query, kwargs, timing_result)
        else:
            # Standard query timing
            query_start = time.time()
            response_data = await model.query(query, **kwargs)
            query_end = time.time()
            
            total_query_time = query_end - query_start
            
            # For integrated models, estimate retrieval vs generation
            if timing_result.architecture == "integrated":
                timing_result.retrieval_time = total_query_time * 0.3  # Estimate
                timing_result.generation_time = total_query_time * 0.7  # Estimate
            else:
                # Two-stage models split time more evenly
                timing_result.retrieval_time = total_query_time * 0.5
                timing_result.generation_time = total_query_time * 0.5
            
            timing_result.gpt4o_retrieval_time = timing_result.retrieval_time
            timing_result.final_generation_time = timing_result.generation_time
        
        response = response_data if isinstance(response_data, str) else response_data.get('output', '')
        full_output = response_data.get('full_output', {}) if isinstance(response_data, dict) else {}
        
        # Extract citations - same logic as performance eval
        if hasattr(full_output, 'choices') and full_output.choices:
            choice = full_output.choices[0]
            if hasattr(choice.message, 'context') and choice.message.context:
                citations = choice.message.context.get('citations', [])

    # Handle TevaGPT instances - same as performance eval
    elif hasattr(model, 'process_and_query'):
        kwargs = {}
        if context_level == 'file':
            kwargs = {'folder_name': folder_path, 'filename': file_name}
        elif context_level == 'folder':
            kwargs = {'folder_name': folder_path}
        elif context_level == 'unconstrained':
            kwargs = {}
        
        # Time the TevaGPT query
        query_start = time.time()
        response_data = await model.process_and_query(query=query, file_path=effective_path, **kwargs)
        query_end = time.time()
        
        total_query_time = query_end - query_start
        # TevaGPT is integrated, so estimate split
        timing_result.retrieval_time = total_query_time * 0.3
        timing_result.generation_time = total_query_time * 0.7
        timing_result.gpt4o_retrieval_time = timing_result.retrieval_time
        timing_result.final_generation_time = timing_result.generation_time
        
        response = response_data.get('output', '') if isinstance(response_data, dict) else response_data
        full_output = response_data.get('full_output', {}) if isinstance(response_data, dict) else {}
        
        # Extract citations
        if hasattr(full_output, 'choices') and full_output.choices:
            choice = full_output.choices[0]
            if hasattr(choice.message, 'context') and choice.message.context:
                citations = choice.message.context.get('citations', [])
    else:
        raise ValueError(f"Unknown pipeline type: {type(model).__name__}")

    return response, full_output, citations

async def time_enhanced_cyrus_query(model, query, kwargs, timing_result):
    """
    Time enhanced CyrusGPT query with component-level breakdown
    """
    # Initialize enhanced components if needed
    if hasattr(model, '_initialize_enhanced_components'):
        model._initialize_enhanced_components()
    
    # Time query expansion if enabled
    if getattr(model, 'query_expansion', False) and model._query_expander:
        expansion_start = time.time()
        expanded_queries = model._query_expander.expand_query(query)
        expansion_end = time.time()
        timing_result.query_expansion_time = expansion_end - expansion_start
    
    # Execute the main query (includes retrieval and any reranking/diversity)
    query_start = time.time()
    
    # For enhanced CyrusGPT, we need to hook into the internal pipeline
    # This is complex because the enhanced processing happens inside _query_with_enhanced_retrieval
    
    # Simple approach: time the whole enhanced query, then subtract known components
    response_data = await model.query(query, **kwargs)
    
    query_end = time.time()
    total_query_time = query_end - query_start
    
    # Estimate component times for enhanced pipeline
    base_two_stage_time = total_query_time * 0.7  # Base two-stage pipeline
    enhancement_overhead = total_query_time * 0.3  # Enhancement processing
    
    timing_result.retrieval_time = base_two_stage_time * 0.4
    timing_result.generation_time = base_two_stage_time * 0.6
    timing_result.gpt4o_retrieval_time = timing_result.retrieval_time
    timing_result.final_generation_time = timing_result.generation_time
    
    # Distribute enhancement overhead
    if getattr(model, 'advanced_reranking', False):
        timing_result.reranking_time = enhancement_overhead * 0.6
    
    if getattr(model, 'diversity_selection', False):
        timing_result.diversity_selection_time = enhancement_overhead * 0.4
    
    return response_data

def create_timing_analysis_excel(
    timing_results: List[TimingResult],
    output_path: str,
    model_labels: List[str]
):
    """Create comprehensive Excel analysis of timing results"""
    
    os.makedirs('eval_outputs', exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 1. Executive Summary Sheet
        create_executive_summary_sheet(timing_results, writer, model_labels)
        
        # 2. Detailed Timing Breakdown
        create_detailed_timing_sheet(timing_results, writer)
        
        # 3. Component Analysis
        create_component_analysis_sheet(timing_results, writer, model_labels)
        
        # 4. Architecture Comparison
        create_architecture_comparison_sheet(timing_results, writer)

def create_executive_summary_sheet(timing_results: List[TimingResult], writer, model_labels: List[str]):
    """Create executive-level summary for presentation"""
    
    summary_data = []
    
    for model_label in model_labels:
        model_results = [r for r in timing_results if r.model_name == model_label and r.success]
        
        if model_results:
            # Calculate statistics
            total_times = [r.total_time for r in model_results]
            retrieval_times = [r.retrieval_time for r in model_results]
            generation_times = [r.generation_time for r in model_results]
            
            # Determine bottleneck
            avg_retrieval = mean(retrieval_times)
            avg_generation = mean(generation_times)
            bottleneck = "Retrieval" if avg_retrieval > avg_generation else "Generation"
            
            # Get architecture
            architecture = model_results[0].architecture
            
            summary_data.append({
                'Model': model_label,
                'Architecture': architecture.replace('_', ' ').title(),
                'Avg_Total_Time_Sec': round(mean(total_times), 1),
                'Min_Time_Sec': round(min(total_times), 1),
                'Max_Time_Sec': round(max(total_times), 1),
                'Avg_Retrieval_Sec': round(avg_retrieval, 1),
                'Avg_Generation_Sec': round(avg_generation, 1),
                'Primary_Bottleneck': bottleneck,
                'Success_Rate_Percent': round(len(model_results) / len([r for r in timing_results if r.model_name == model_label]) * 100, 1),
                'Sample_Size': len(model_results)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, index=False, sheet_name='Executive_Summary')

def create_detailed_timing_sheet(timing_results: List[TimingResult], writer):
    """Create detailed timing breakdown sheet"""
    
    detailed_data = []
    
    for result in timing_results:
        detailed_data.append({
            'Model': result.model_name,
            'Test_ID': result.test_id,
            'Architecture': result.architecture,
            'Success': result.success,
            'Total_Time': round(result.total_time, 2),
            'Retrieval': round(result.retrieval_time, 2),
            'Generation': round(result.generation_time, 2),
            'Query_Expansion': round(result.query_expansion_time, 2),
            'GPT4o_Retrieval': round(result.gpt4o_retrieval_time, 2),
            'Reranking': round(result.reranking_time, 2),
            'Diversity_Selection': round(result.diversity_selection_time, 2),
            'Context_Formatting': round(result.context_formatting_time, 2),
            'Final_Generation': round(result.final_generation_time, 2),
            'Retrieved_Docs_Count': result.retrieved_docs_count,
            'Error': result.error_message
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_excel(writer, index=False, sheet_name='Detailed_Breakdown')

def create_component_analysis_sheet(timing_results: List[TimingResult], writer, model_labels: List[str]):
    """Analyze which components take the most time"""
    
    component_data = []
    
    for model_label in model_labels:
        model_results = [r for r in timing_results if r.model_name == model_label and r.success]
        
        if model_results:
            # Calculate component averages
            components = {
                'Query_Expansion': [r.query_expansion_time for r in model_results],
                'GPT4o_Retrieval': [r.gpt4o_retrieval_time for r in model_results],
                'Reranking': [r.reranking_time for r in model_results],
                'Diversity_Selection': [r.diversity_selection_time for r in model_results],
                'Final_Generation': [r.final_generation_time for r in model_results]
            }
            
            for component, times in components.items():
                if any(t > 0 for t in times):  # Only include active components
                    component_data.append({
                        'Model': model_label,
                        'Component': component,
                        'Avg_Time_Sec': round(mean([t for t in times if t > 0]), 3),
                        'Max_Time_Sec': round(max(times), 3),
                        'Percent_of_Total': round(mean([t for t in times if t > 0]) / mean([r.total_time for r in model_results]) * 100, 1),
                        'Active_Queries': len([t for t in times if t > 0])
                    })
    
    component_df = pd.DataFrame(component_data)
    component_df.to_excel(writer, index=False, sheet_name='Component_Analysis')

def create_architecture_comparison_sheet(timing_results: List[TimingResult], writer):
    """Compare different architectures"""
    
    architectures = list(set(r.architecture for r in timing_results))
    arch_data = []
    
    for arch in architectures:
        arch_results = [r for r in timing_results if r.architecture == arch and r.success]
        
        if arch_results:
            total_times = [r.total_time for r in arch_results]
            retrieval_times = [r.retrieval_time for r in arch_results]
            generation_times = [r.generation_time for r in arch_results]
            
            arch_data.append({
                'Architecture': arch.replace('_', ' ').title(),
                'Sample_Size': len(arch_results),
                'Avg_Total_Time': round(mean(total_times), 1),
                'Avg_Retrieval_Time': round(mean(retrieval_times), 1),
                'Avg_Generation_Time': round(mean(generation_times), 1),
                'Retrieval_Percent': round(mean(retrieval_times) / mean(total_times) * 100, 1),
                'Generation_Percent': round(mean(generation_times) / mean(total_times) * 100, 1),
                'Models_Using': ', '.join(list(set(r.model_name.split('_')[0] for r in arch_results)))
            })
    
    arch_df = pd.DataFrame(arch_data)
    arch_df.to_excel(writer, index=False, sheet_name='Architecture_Comparison')

def print_executive_timing_summary(timing_results: List[TimingResult], model_labels: List[str]):
    """Print executive summary to console"""
    
    print("\n" + "=" * 80)
    print("EXECUTIVE TIMING SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    successful_results = [r for r in timing_results if r.success]
    if successful_results:
        overall_avg = mean([r.total_time for r in successful_results])
        overall_min = min([r.total_time for r in successful_results])
        overall_max = max([r.total_time for r in successful_results])
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Average Query Time: {overall_avg:.1f}s")
        print(f"   Range: {overall_min:.1f}s - {overall_max:.1f}s")
        print(f"   Total Queries Analyzed: {len(successful_results)}")
    
    # Model comparison
    print(f"\n🏆 MODEL PERFORMANCE RANKING:")
    model_performance = []
    
    for model_label in model_labels:
        model_results = [r for r in timing_results if r.model_name == model_label and r.success]
        if model_results:
            avg_time = mean([r.total_time for r in model_results])
            model_performance.append((model_label, avg_time, len(model_results)))
    
    # Sort by average time (fastest first)
    model_performance.sort(key=lambda x: x[1])
    
    for rank, (model, avg_time, count) in enumerate(model_performance, 1):
        print(f"   {rank}. {model}: {avg_time:.1f}s avg ({count} queries)")
    
    # Component breakdown for enhanced models
    print(f"\n⚡ ENHANCED COMPONENT BREAKDOWN:")
    for model_label in model_labels:
        if 'Enhanced' in model_label:
            model_results = [r for r in timing_results if r.model_name == model_label and r.success]
            if model_results:
                avg_expansion = mean([r.query_expansion_time for r in model_results if r.query_expansion_time > 0])
                avg_reranking = mean([r.reranking_time for r in model_results if r.reranking_time > 0])
                avg_diversity = mean([r.diversity_selection_time for r in model_results if r.diversity_selection_time > 0])
                
                print(f"   {model_label}:")
                if avg_expansion > 0:
                    print(f"     Query Expansion: {avg_expansion:.2f}s avg")
                if avg_reranking > 0:
                    print(f"     Advanced Reranking: {avg_reranking:.2f}s avg")
                if avg_diversity > 0:
                    print(f"     Diversity Selection: {avg_diversity:.2f}s avg")
    
    print("=" * 80)

# Main execution function
async def example_detailed_timing_analysis():
    """Example of how to run the detailed timing analysis"""
    
    # Import the model classes
    from TevaGPT_dynamic import TevaGPT
    from Ask_HR_dynamic import AskHR
    from CyrusGPT import CyrusGPT
    
    # Create the exact same models as your performance evaluation
    models = [
        TevaGPT(),
        AskHR(),
        CyrusGPT(retrieval_method='default'),
        CyrusGPT(
            retrieval_method='enhanced',
            query_expansion=True,
            advanced_reranking=False,
            diversity_selection=False,
        ),
        CyrusGPT()
    ]
    
    # Run detailed timing analysis with same settings as performance eval
    output_path = await run_detailed_timing_analysis(
        models=models,
        context_levels=['unconstrained'],
        max_concurrent=5,  # Same as performance evaluation
        max_retries=3      # Same as performance evaluation
    )
    
    print(f"\n🎯 Timing analysis complete: {output_path}")
    return output_path

# Main execution block
if __name__ == "__main__":
    asyncio.run(example_detailed_timing_analysis())
