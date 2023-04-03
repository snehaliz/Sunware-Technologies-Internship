#imports
import os, sys
import traceback
import pandas as PD
import numpy as NP
import logging

file_name = __file__ + '.txt'
logging.basicConfig(
  level = logging.DEBUG ,
  filename = file_name ,
  filemode = 'w',
  format = '%(asctime)s - %(message)s'
  )

#_text = 'I bought a Toyota Corala in January last year. It works well..'
#_questions = [
  #'What did the person buy?', #returns 'Toyota Corala'
  #'When did he buy?', #returns 'January last year'
  #'What is working?', #returns 'Toyota Corala'
  #]

#First, I tried the texts all with the distilbert-base-uncased-distilled-squad model to see what the answers were. Then, with the example of
#buying a ticket to LA, I tried the different models. 

#_text = '''I'd like to buy a house in California.'''
#_questions = [
  #'What did the person buy?', #returns 'a house'
  #'Where did he buy it?', #returns 'California'
  #]

#_text = '''I am looking with JPL headphones with Bluetooth support.'''
#_questions = [
  #'What is the person looking for?', #returns 'JPL headphones with Bluetooth support'
  #'What kind of support should the headphones have?', #returns 'Bluetooth'
  #]

_text = '''I am looking for a ticket to LA this weekend.'''
_questions = [
  'What is the person looking for?', #returns 'a ticket to LA'
  'Where does the person want to go?', #returns 'LA'
  'When does the person want to go to LA?',#returns 'weekend'
  ]



#Different model paths are run for the question 'What is the person looking for?'
_model_path = 'anas-awadalla/gpt2-medium-span-head-few-shot-k-32-finetuned-squad-seed-2' # did not return anything
#_model_path = 'hf-internal-testing/tiny-random-GPTJForQuestionAnswering' #returns 'ticket to LA'
#_model_path = 'ydshieh/tiny-random-gptj-for-question-answering' #returns 'to LA'
#_model_path = 'anas-awadalla/gpt2-large-span-head-finetuned-squad' # did not return anything
#_model_path = 'deepset/roberta-base-squad2-distilled' #For first question of ticket to LA, returns "a ticket to LA this weekend"
#_model_path = 'distilbert-base-uncased-distilled-squad' #returns 'a ticket to LA'

def main():
  '''Main entry point'''
  try:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipelines
    os.environ['TRANSFORMERS_CACHE'] = 'D:\\python\\transformers_cache'
    
    #nlp = pipelines.pipeline( 'question-answering', model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-large-squad2") , tokenizer  =  AutoTokenizer.from_pretrained("deepset/roberta-large-squad2") )
    nlp = pipelines.pipeline( 'question-answering', model = AutoModelForQuestionAnswering.from_pretrained(_model_path ) , tokenizer  =  AutoTokenizer.from_pretrained( _model_path ) )
    qa_input = {}
    qa_input['question'] = _questions[2]
    qa_input['context'] = _text

    answer = nlp(qa_input)
    print(answer)
    #print(answer)
  except:
    logging.exception("Exception in the code.")

  return


'''
Required for all python programs.
'''
if __name__ == '__main__':
  print('Starting the QA tool.')
  main()
  print('Done\nPress <Enter> to exit.')
  input()

