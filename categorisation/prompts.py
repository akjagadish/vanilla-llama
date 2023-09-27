def retrieve_prompt(model, version):

    instructions = {}

    # llama
    num_dim, num_data = 3, 8
    llama_prompt_v0 = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" The following are {str(num_data)} input-target pairs generated for one such classification problem:\n"\
                        "x=["
    
    instructions['llama'] = {}
    instructions['llama']['v0'] = llama_prompt_v0

    # gpt3
    gpt3_prompt_v0 = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" Please generate a list of {str(num_data)} input-target pairs using the following template for each row:\n"\
                        f"- [x1, x2, x3], y"
    
    gpt3_prompt_v1 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                    " You can choose any naturalistic decision function for the mapping from input to target.  \n\n"\
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y"
    
    instructions['gpt3'] = {}
    instructions['gpt3']['v0'] = gpt3_prompt_v0
    instructions['gpt3']['v1'] = gpt3_prompt_v1

    # gpt4
    gpt4_prompt_v0 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y" ## got code to generate output once but otherwise consistent 
    
    gpt4_prompt_v1 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                    " You can choose any naturalistic decision function for the mapping from input to target. \n\n"\
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y"\
                    f" Do not generate any text but just provide the input-target pairs." ## moved this line for pre-prompt
    
    gpt4_prompt_v2 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                    " For the mapping from input to target, a wide range of naturalistic decision functions can be chosen."\
                    " These functions may encompass complex mathematical operations, linear or non-linear functions, or arbitrary rule-based systems."\
                    " The selected function should be representative of patterns or rules that may exist in real-world categorization tasks. \n\n" \
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y"
                    
    features = ['shape', 'size', 'colour']
    gpt4_prompt_v3 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of objects and their category labels."\
                    f" Each object is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    " These feature values (rounded to 2 decimals) range continuously between 0 and 1."\
                    " Each feature should follow a distribution that describes the values they take in the real world. "\
                    " The category label can take the values A or B and should be predictable from the feature values of the object."\
                    " For the mapping from object features to the category label, you can choose any naturalistic function that is"\
                    " representative of patterns or rules that may exist in real-world tasks. \n\n"\
                    f" Please generate a list of {str(num_data)} objects with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"
    
    instructions['gpt4'] = {}
    instructions['gpt4']['v0'] = gpt4_prompt_v0
    instructions['gpt4']['v1'] = gpt4_prompt_v1
    instructions['gpt4']['v2'] = gpt4_prompt_v2
    instructions['gpt4']['v3'] = gpt4_prompt_v3


    # claude
    features = ['shape', 'size', 'color']
    claude_prompt_v0 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of objects and their category labels."\
                    f" Each object is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    " These feature values (rounded to 2 decimals) range continuously between 0 and 1."\
                    " Each feature should follow a distribution that describes the values they take in the real world. "\
                    " The category label can take the values A or B and should be predictable from the feature values of the object."\
                    " For the mapping from object features to the category label, you can choose any naturalistic function that is"\
                    " representative of patterns or rules that may exist in real-world tasks. \n\n"\
                    f" Please generate a list of {str(num_data)} objects with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"

    claude_prompt_v1 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of objects and their category labels."\
                    f" Each object is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    " These feature values (rounded to 2 decimals) range continuously between 0 and 1."\
                    " Each feature should follow a distribution that describes the values they take in the real world. "\
                    " The category label can take the values A or B and should be predictable from the feature values of the object."\
                    " \n\n"\
                    f" Please generate a list of {str(num_data)} objects with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"
    
    instructions['claude'] = {}
    instructions['claude']['v0'] = claude_prompt_v0
    instructions['claude']['v1'] = claude_prompt_v1
    
    return instructions[model][version]