from nada_dsl import *
import os

def nada_main():


    party1 = Party(name="Party1")

    my_int1 = SecretInteger(Input(name="my_int1", party=party1))

    my_int2 = SecretInteger(Input(name="my_int2", party=party1))
    

   


    current_directory = os.getcwd()
    input_file = os.path.join(current_directory, "output.json")

    import json
    
    with open(input_file,"r") as file:
         input_data=json.load(file) 
         
    
    result=int(input_data["result"])
    

    
   
    if result == 0:
         return [Output(my_int1+my_int2, "my_output", party1)]
         
    else:
         return [Output(my_int1-my_int2, "my_output", party1)]