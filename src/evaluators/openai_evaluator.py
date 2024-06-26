
import os

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI

class OpenAIEvaluator(Evaluator):
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 api_key: str = None,
                 true_answer: str = None,
                 question_asked: str = None):
        """
        :param model_name: The name of the model.
        :param api_key: The API key for OpenAI. Default is None.
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.true_answer = true_answer
        self.question_asked = question_asked

        if (api_key is None) and (not os.getenv('OPENAI_API_KEY')):
            raise ValueError("Either api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        self.evaluator = ChatOpenAI(model=self.model_name,
                                    temperature=0,
                                    openai_api_key=self.api_key)

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])