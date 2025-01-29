from dataclasses import dataclass
from typing import List, Dict
import asyncio
import time
import statistics
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import logging

from src.scripts.query import QueryProcessor


@dataclass
class ModelTestResult:
    """Stores test results for a single model"""
    model_name: str
    avg_latency_ms: float
    hallucination_rate: float
    response_consistency: float
    multilingual_score: Dict[str, float]
    responses: Dict[str, List[Dict[str, str]]]


class LLMTestingSuite:
    def __init__(
            self,
            models: List[str],
            test_queries: Dict[str, List[str]],
            ground_truth: Dict[str, List[str]],
            config_path: str
    ):
        self.models = models
        self.test_queries = test_queries  # Dict with 'pt' and 'es' keys
        self.ground_truth = ground_truth
        self.config_path = config_path
        self.logger = logging.getLogger("llm_testing")

    async def measure_latency(
            self,
            model: str,
            language: str,
            n_samples: int = 50
    ) -> float:
        """Measures average response latency for a model"""
        latencies = []
        queries = self.test_queries[language][:n_samples]

        for query in tqdm(queries, desc=f"Testing latency for {model}"):
            start_time = time.time()
            try:
                query_processor = QueryProcessor(self.config_path)
                query_processor.llm_provider.model_name = model

                await query_processor.process_query(
                    query_text=query,
                    language=language
                )
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                self.logger.error(f"Error during latency test: {str(e)}")
                continue

        return statistics.mean(latencies) if latencies else float('inf')

    async def measure_hallucination_rate(
            self,
            model: str,
            language: str,
            n_samples: int = 50
    ) -> float:
        """Estimates hallucination rate using ground truth comparison"""
        hallucinations = 0
        queries = self.test_queries[language][:n_samples]
        ground_truths = self.ground_truth[language][:n_samples]

        for query, truth in tqdm(zip(queries, ground_truths), desc=f"Testing hallucination for {model}"):
            try:
                # Configurar o modelo no query processor
                query_processor = QueryProcessor(self.config_path)
                query_processor.llm_provider.model_name = model

                response = await query_processor.process_query(
                    query_text=query,
                    language=language
                )

                # Simple text similarity check - could be enhanced with better metrics
                response_lower = response.lower()
                truth_lower = truth.lower()

                # Verifica se pelo menos uma das informações verdadeiras está na resposta
                if not any(fact.lower() in response_lower for fact in truth_lower.split('.')):
                    hallucinations += 1
            except Exception as e:
                self.logger.error(f"Error during hallucination test: {str(e)}")
                hallucinations += 1  # Consider failed responses as hallucinations
                continue

        return hallucinations / len(queries)

    async def measure_consistency(
            self,
            model: str,
            language: str,
            n_pairs: int = 25
    ) -> float:
        """Measures response consistency for similar queries"""
        consistency_scores = []

        # Generate similar query pairs
        similar_pairs = self._generate_similar_pairs(
            self.test_queries[language],
            n_pairs
        )

        for query1, query2 in tqdm(similar_pairs, desc=f"Testing consistency for {model}"):
            try:
                # Configurar o modelo no query processor
                query_processor = QueryProcessor(self.config_path)
                query_processor.llm_provider.model_name = model

                response1 = await query_processor.process_query(
                    query_text=query1,
                    language=language
                )
                response2 = await query_processor.process_query(
                    query_text=query2,
                    language=language
                )

                # Calculate similarity score between responses
                similarity = self._calculate_response_similarity(response1, response2)
                consistency_scores.append(similarity)
            except Exception as e:
                self.logger.error(f"Error during consistency test: {str(e)}")
                continue

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    def _generate_similar_pairs(
            self,
            queries: List[str],
            n_pairs: int
    ) -> List[tuple]:
        """Generates pairs of similar queries"""
        pairs = []
        for _ in range(n_pairs):
            query = np.random.choice(queries)
            words = query.split()
            if len(words) > 1:
                # Substitui uma palavra aleatória por um sinônimo ou palavra similar
                idx = np.random.randint(len(words))

                # Dicionário simples de substituições
                replacements = {
                    'como': 'qual',
                    'qual': 'como',
                    'procedimento': 'processo',
                    'processo': 'procedimento',
                    'automatizar': 'preencher',
                    'preencher': 'automatizar'
                }

                word = words[idx].lower()
                if word in replacements:
                    words[idx] = replacements[word]
                    similar_query = ' '.join(words)
                    pairs.append((query, similar_query))

        return pairs[:n_pairs]  # Garante que temos exatamente n_pairs

    def _calculate_response_similarity(
            self,
            response1: str,
            response2: str
    ) -> float:
        """Calculates similarity between two responses"""
        # Simple Jaccard similarity
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def evaluate_model(self, model: str) -> ModelTestResult:
        """Runs all tests for a single model"""
        results = {}
        responses = {'pt': [], 'es': []}

        # Test both languages
        for language in ['pt', 'es']:
            self.logger.info(f"Testing {model} in {language}")

            # Criar uma nova instância do QueryProcessor para este modelo/idioma
            query_processor = QueryProcessor(self.config_path)
            query_processor.llm_provider.model_name = model

            # Coletar todas as respostas primeiro
            queries = self.test_queries[language][:5]  # Usando 5 amostras
            for query in tqdm(queries, desc=f"Collecting responses for {model} - {language}"):
                try:
                    response = await query_processor.process_query(
                        query_text=query,
                        language=language
                    )
                    responses[language].append({
                        'query': query,
                        'response': response,
                        'ground_truth': self.ground_truth[language][queries.index(query)]
                    })
                except Exception as e:
                    self.logger.error(f"Error collecting response: {str(e)}")
                    responses[language].append({
                        'query': query,
                        'response': f"ERROR: {str(e)}",
                        'ground_truth': self.ground_truth[language][queries.index(query)]
                    })

            # Calcular métricas usando as respostas coletadas
            results[language] = {
                'latency': await self.measure_latency(model, language, n_samples=5),
                'hallucination_rate': await self.measure_hallucination_rate(
                    model,
                    language,
                    n_samples=5
                ),
                'consistency': await self.measure_consistency(model, language, n_pairs=5)
            }

        # Calculate overall metrics
        avg_latency = statistics.mean(
            [results[lang]['latency'] for lang in ['pt', 'es']]
        )

        # Calculate multilingual score
        multilingual_scores = {}
        for language in ['pt', 'es']:
            lang_metrics = results[language]
            normalized_score = (
                    (1 - lang_metrics['hallucination_rate']) * 0.4 +
                    lang_metrics['consistency'] * 0.4 +
                    (1 - (lang_metrics['latency'] / 5000)) * 0.2
            )
            multilingual_scores[language] = normalized_score

        return ModelTestResult(
            model_name=model,
            avg_latency_ms=avg_latency,
            hallucination_rate=statistics.mean(
                [results[lang]['hallucination_rate'] for lang in ['pt', 'es']]
            ),
            response_consistency=statistics.mean(
                [results[lang]['consistency'] for lang in ['pt', 'es']]
            ),
            multilingual_score=multilingual_scores,
            responses=responses
        )

    async def run_all_tests(self) -> tuple[pd.DataFrame, Dict]:
        """Runs all tests for all models and returns results as DataFrame and detailed responses"""
        all_results = []
        all_responses = {}

        for model in self.models:
            self.logger.info(f"Testing model: {model}")
            try:
                result = await self.evaluate_model(model)
                all_results.append(result)
                all_responses[model] = result.responses
            except Exception as e:
                self.logger.error(f"Error testing {model}: {str(e)}")

        # Convert results to DataFrame
        df_data = []
        for result in all_results:
            row = {
                'Model': result.model_name,
                'Avg Latency (ms)': round(result.avg_latency_ms, 2),
                'Hallucination Rate (%)': round(result.hallucination_rate * 100, 2),
                'Response Consistency': round(result.response_consistency, 2),
                'PT Score': round(result.multilingual_score['pt'], 2),
                'ES Score': round(result.multilingual_score['es'], 2),
                'Overall Score': round(
                    (result.multilingual_score['pt'] +
                     result.multilingual_score['es']) / 2,
                    2
                )
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Salvar resultados detalhados em JSON
        detailed_results = {
            'metrics': df.to_dict(orient='records'),
            'responses': all_responses
        }

        with open('llm_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        # Salvar respostas em um formato mais legível
        with open('llm_responses.txt', 'w', encoding='utf-8') as f:
            for model in all_responses:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Model: {model}\n")
                f.write(f"{'=' * 80}\n\n")

                for lang in ['pt', 'es']:
                    f.write(f"\nLanguage: {lang}\n")
                    f.write(f"{'-' * 80}\n\n")

                    for item in all_responses[model][lang]:
                        f.write(f"Query: {item['query']}\n")
                        f.write(f"Ground Truth: {item['ground_truth']}\n")
                        f.write(f"Response: {item['response']}\n")
                        f.write(f"{'-' * 40}\n\n")

        return df, all_responses


async def main():
    # Example usage
    test_queries = {
        'pt': [
            'Como automatizar o preenchimento dos códigos?',
            'Qual o procedimento para calcular IRRF?',
            'Como configurar o campo U_TX_IndNat?',
            'Quais são os requisitos para carregamento automático?',
            'Como os valores são transportados para o registro N620-IRPJ?'
        ],
        'es': [
            '¿Cómo automatizar el llenado de los códigos?',
            '¿Cuál es el procedimiento para calcular IRRF?',
            '¿Cómo configurar el campo U_TX_IndNat?',
            '¿Cuáles son los requisitos para la carga automática?',
            '¿Cómo se transportan los valores al registro N620-IRPJ?'
        ]
    }

    ground_truth = {
        'pt': [
            'Para automatizar é necessário informar as opções 01, 02 ou 03 no campo U_TX_IndNat.',
            'No documento de saída calcule IRRF e CSLL usando tipos nativos do SAP.',
            'No campo U_TX_IndNat do cadastro de parceiro de negócio informe as opções: 01, 02 ou 03.',
            'É necessário atender dois requisitos: configurar U_TX_IndNat e usar tipos nativos do SAP para IRRF e CSLL.',
            'Os valores são transportados para os registros N620-IRPJ, códigos 14, 15, 16 de acordo com a opção no cadastro.'
        ],
        'es': [
            'Para automatizar es necesario informar las opciones 01, 02 o 03 en el campo U_TX_IndNat.',
            'En el documento de salida calcule IRRF y CSLL usando tipos nativos de SAP.',
            'En el campo U_TX_IndNat del registro de socio comercial, informe las opciones: 01, 02 o 03.',
            'Es necesario cumplir dos requisitos: configurar U_TX_IndNat y usar tipos nativos de SAP para IRRF y CSLL.',
            'Los valores se transportan a los registros N620-IRPJ, códigos 14, 15, 16 según la opción en el registro.'
        ]
    }

    models = [
        'mistral',
        'neural-chat',
        'zephyr',
        'starling-lm',
        'openhermes'
    ]

    test_suite = LLMTestingSuite(
        models=models,
        test_queries=test_queries,
        ground_truth=ground_truth,
        config_path="config.yaml"
    )

    results_df, responses = await test_suite.run_all_tests()
    print("\nTest Results:")
    print(results_df.to_string())


if __name__ == "__main__":
    asyncio.run(main())