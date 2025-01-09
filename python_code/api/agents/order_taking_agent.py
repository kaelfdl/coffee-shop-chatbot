import os
import json
import dotenv
from openai import OpenAI
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output
dotenv.load_dotenv()


class OrderTakingAgent():
    def __init__(self, recommendation_agent):
        self.client = OpenAI(
            api_key = os.getenv("RUNPOD_TOKEN"),
            base_url = os.getenv("RUNPOD_CHATBOT_URL")
        )
        self.model_name = os.getenv("MODEL_NAME")

        self.recommendation_agent = recommendation_agent

    def get_response(self, messages):
        messages = deepcopy(messages)


        system_prompt = """
        You are a customer support bot for a coffee shop called "Harvest Roast"

        Here is the menu for this coffee shop.

        
        Cappuccino - $4.50
        Latte - $4.75
        Chocolate Chip Biscotti - $2.50
        Espresso shot - $2.00
        Hazelnut Biscotti - $2.75
        Chocolate Croissant - $3.75
        Dark chocolate (Drinking Chocolate) - $5.00
        Cranberry Scone - $3.50
        Croissant - $3.25
        Almond Croissant - $4.00
        Ginger Biscotti - $2.50
        Scottish Cream Scone - $3.25
        Chocolate syrup - $1.50
        Hazelnut syrup - $1.50
        Carmel syrup - $1.50
        Sugar Free Vanilla syrup - $1.50
        Dark chocolate (Packaged Chocolate) - $3.00

        Things to NOT DO:
        * Don't ask how to pay by cash or card.
        * Don't tell the user to go to the counter
        * Don't tell the user to go to the place to get the order

        Your task is as follows:
        1. Take the user's order
        2. Validate that all their items are in the menu
        3. If an item is not in the menu, let the user know, and repeat back the remaining valid order
        4. Ask them if they need anything else.
        5. If they do then repeat starting from step 3
        6. If they don't want anything else. Using the "order" object that is in the output. Make sure to hit all three points
            1. List down all the items and their prices
            2. Calculate the total
            3. Thank the user for the order and close the conversation with no more questions

        The user message will contain a section called memory. This section will contain the following:
        "order"
        "step number"
        please utilize this information to determine the next step in the process.

        Produce the following output without any additions, not a single letter outside of the structure below.
        Your output should be in the json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly.
        {
            "chain of thought": Write down your critical thinking about what is the maximum task number the user is on right now. Then write down your critical thinking about the user input and its relation to the coffee shop process. Then write down your thinking about how you should respond in the response parameter taking into consideration the things to NOT DO section. and focus on the things that you should not do.
            "step_number": Determine which task you are on based on the conversation
            "order": This is going to be a list of jsons like so. [{"item": put the item name, "quantity": put the number that the user wants from this item, "price": put the total price of this item}]
            "response": write the response to the user
        }
        """

        last_order_taking_status = ""

        asked_recommendation_before = False
        for message_index in range(len(messages) - 1, 0, -1):
            message = messages[message_index]

            agent_name = message.get("memory", {}).get("agent", "")
            if message["role"] == "assistant" and agent_name == "order_taking_agent":
                step_number = message["memory"]["step_number"]
                order = message["memory"]["order"]
                asked_recommendation_before = message["memory"]["asked_recommendation_before"]
                last_order_taking_status = f"""
                step_number: {step_number}
                order: {order}
                """

        messages[-1]["content"] = last_order_taking_status + " \n" + messages[-1]["content"]
        input_messages = [{"role": "system", "content": system_prompt}] + messages
        chatbot_response = get_chatbot_response(self.client, self.model_name, input_messages)
        chatbot_response = double_check_json_output(self.client, self.model_name, chatbot_response)

        output = self.postprocess(chatbot_response, messages, asked_recommendation_before)
        return output


    def postprocess(self, output, messages, asked_recommendation_before):
        output = json.loads(output)

        if type(output["order"]) == str:
            output["order"] = json.loads(output["order"])
        
        response = output["response"]
        
        if not asked_recommendation_before and len(output["order"]) > 0:
            recommendation_output = self.recommendation_agent.get_recommendations_from_order(messages, output["order"])
            response = recommendation_output["content"]
            asked_recommendation_before = True
        
        dict_output = {
            "role": "assistant",
            "content": response,
            "memory": {
                "agent": "order_taking_agent",
                "step_number": output["step_number"],
                "order": output["order"],
                "asked_recommendations_before": asked_recommendation_before
            }
        }
        return dict_output