from small_llm import SmallLLM
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Chatbot...")
    try:
        # Initialize and load the model
        model = SmallLLM()
        model.load_model()
        logger.info("Model loaded successfully.")

        print("Chatbot initialized. Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == 'quit':
                    logger.info("User requested to quit. Exiting chatbot.")
                    break
                
                if not user_input:
                    print("Bot: I'm sorry, but I didn't receive any input. Could you please try again?")
                    continue

                # Get the model's response
                response = model.predict(user_input)
                print(f"Bot: {response}")

                # Optional: Add feedback mechanism
                feedback = input("Was this response helpful? (y/n): ").strip().lower()
                if feedback == 'y':
                    model.add_feedback(user_input, response, reward=1)
                elif feedback == 'n':
                    model.add_feedback(user_input, response, reward=-1)
                
            except Exception as e:
                logger.error(f"Error during chat interaction: {str(e)}")
                logger.error(traceback.format_exc())
                print("Bot: I apologize, but I encountered an error. Let's try again.")

    except Exception as e:
        logger.error(f"Error initializing or running chatbot: {str(e)}")
        logger.error(traceback.format_exc())
        print("An error occurred while setting up the chatbot. Please check the logs for more information.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        logger.critical(traceback.format_exc())
        print("A critical error occurred. The chatbot has to shut down. Please check the logs for more information.")