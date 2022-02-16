import os
from datetime import datetime
import time    
    
    # Saahiti Dictionary Function
   
    
def create_param_dictionaries(all_parameters):
    # Input: all_parameters - a dictionary of parameters, including some parameter arrays for hyperparameter searches
    # Output: parameter_dicts - a list of dictionaries, where each dictionary defines parameters for one run of the model
    # Function Description: convert the input dictionary (which includes lists of parameter searches) to a list of dictionaries
                            # each defining a run of the model.  Should have all permutations of parameters. ie. if there are 3
                            # lists included of 2,3, and 5 parameter options respectively, the output list should include 
                            #2x3x5=30 dictionaries
    
    parameter_dicts = []
    lists = {}
    
    # Find and record all lists (if any)
    for key in all_parameters.keys():
        if type(all_parameters[key]) is list:
            lists.update({key: all_parameters[key]})
    
    # If there were none, move on
    if lists == {}:
        parameter_dicts.append(all_parameters)
    
    else:
    
        # Determine how many combinations there are (multiply lengths of lists)
        copies = 1
        for key in lists.keys():
            copies *= len(lists[key])
            
        # Copy the dict this many times into the big list
        for i in range(copies):
            parameter_dicts.append(all_parameters.copy())

        # Count keeps track of how many repetitions of each value to distribute
        count = int(copies)
        
        # For each key, divide count by length of list to determine how many times
        # each value repeats
        for key in lists.keys():
            count /= len(lists[key])
            
            # Index for which dictionary/variation is being altered
            curr = 0
            
            # Determine how many times the set of values is repeated
            # i.e. reps * count should equal copies
            reps = copies / (len(lists[key])*count)
            for rep in range(int(reps)):
            
                # Go through values in the list
                for val in lists[key]:

                    # Enter the value count number of times
                    for j in range(int(count)):
                        parameter_dicts[curr][key] = val
                        curr += 1

#   print parameter_dicts

    return parameter_dicts

# all_parameters = {"a": 0,
#                  "b": [11,33,55],
#                  "c": 0,
#                  "d": [22,222],
#                  "e": 0,
#                  "f": [8,88,888,8888]}
#parameter_dicts = create_param_dictionaries(all_parameters)



def convert2htmlrow(run, outcomes, node, port, stage_names):
    
    data_folder = run[0]
    experiment = '<td class="tg-0lax">' + run[1] + '</td>\n'
    date = '<td class="tg-0lax">' + datetime.utcfromtimestamp(int('1'+run[1].split('_')[-1])).strftime('%m-%d %H:%M') + '</td>\n'
    model_id = '<td class="tg-0lax">' + str(run[2]) + '</td>\n'
    stage = '<td class="tg-0lax">' + str(run[3]) + '</td>\n'
    
    outcome_str = ''
    #print(outcomes)
    #print(run)
    
    
    
    for out_ind in range(len(outcomes)):
        o = outcomes[out_ind]
        
        hyper = node + ':' + port + '/edit/clip-realspeech/projects/vid_game/data/' + data_folder + '/log/'+run[1] + \
        '/'+stage_names[out_ind] + '.'+str(run[2])+ '.' + str(run[3]) + '.log'
        
        if o == None:
            outcome_str += '<td class="tg-0lax"></td>\n'
        elif o == 1:
            outcome_str += '<td class="tg-og4q"><a href="' + hyper + '">Fail</a></td>\n'
        elif o == 2:
            outcome_str += '<td class="tg-og4q"><a href="' + hyper + '">Fail</a></td>\n'
        elif o  == 0:
            outcome_str += '<td class="tg-fd62"><a href="' + hyper + '">Success</a></td>\n'
        elif o == -999:
            outcome_str += '<td class="tg-e76x"><a href="' + hyper + '">Running</a></td>\n'
            
        else:
            print(o)
            raise(KeyError)
        
    return '<tr>\n' + experiment + date + model_id + stage + outcome_str + '</tr>\n' 

def generate_log_table(node, port, experiment_folder):

    log_folder = '../../data/' + experiment_folder + '/log'
    os.listdir(log_folder)
    directory_list = os.listdir(log_folder)
    experiments = []
    for x in directory_list:
        if os.path.isdir(log_folder + '/' + x):
            experiments.append(x)

    stage_names = ['acousticgame_pretrain_network', 'acousticgame_pretrain_validation', 'acousticgame_calculate_ewc_coeffs', 
                   'acousticgame_run_abx', 'acousticgame_train', 'game_process_experiment', 'acousticgame_test'] #run_abxtrain-1 ADD
    stage_names = ['acousticgame_pretrain_network', 'acousticgame_pretrain_validation', 'acousticgame_calculate_ewc_coeffs', 
                   'acousticgame_run_abxpretrain-1', 'acousticgame_train', 'game_process_experiment', 'acousticgame_run_abxtrain-1']#test-1         
    log_outcomes = {}
    unix_codes = []
    for exp in experiments:
        logfiles_names = [f for f in os.listdir(log_folder + '/' + exp) if not os.path.isdir(log_folder + '/' + exp + '/' + f)]
        for l in logfiles_names:

            stage, model, run, log = l.split('.')
            unix_code = int(exp.split('_')[-1])
            f = open(log_folder + '/' + exp + '/' + l)
            lines = f.readlines()
            if lines[-1][:14]=='# Finished at ':            
                return_value = lines[-1].split(' ')[-1].strip()
            else:
                return_value=-999

            if unix_code not in unix_codes:
                log_outcomes[unix_code] = {(experiment_folder, exp, model, run): [None, None, None, None, None, None, None]}
                unix_codes.append(unix_code)




            stage_index = stage_names.index(stage)

            #print(log_outcomes)
            if (experiment_folder, exp, model, run) not in log_outcomes[unix_code].keys():
                    log_outcomes[unix_code][(experiment_folder, exp, model, run)] = [None, None, None, None, None, None, None]

            if log_outcomes[unix_code][(experiment_folder, exp, model, run)][stage_index] is not None:
                #print(log_outcomes[unix_code])
                raise(KeyError)
            else:
                log_outcomes[unix_code][(experiment_folder, exp, model, run)][stage_index] = int(return_value)

   # for d in log_outcomes.keys():
  #      print(d)
  #      print(log_outcomes[d])

    unix_codes.sort(reverse=True)

    #<td class="tg-0lax"></td>
    table_str = ''
    for row in unix_codes:
        for run, outcomes in log_outcomes[row].items():
            table_str += convert2htmlrow(run, outcomes, node, port, stage_names)
            
    experiment_title = ' '.join([e for e in experiment_folder.split('_')])   
    html_prefix = '<h1>' + experiment_title + '</h1><style type="text/css">.tg  {border-collapse:collapse;border-spacing:0;}.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;  overflow:hidden;padding:10px 5px;word-break:normal;}.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}.tg .tg-e76x{background-color:#fe996b;text-align:left;vertical-align:top}.tg .tg-0lax{text-align:left;vertical-align:top}.tg .tg-og4q{background-color:#fd6864;text-align:left;vertical-align:top}.tg .tg-fd62{background-color:#32cb00;text-align:left;vertical-align:top}</style><table class="tg"><tbody>'
    html_firstrow = '<tr>    <td class="tg-0lax">Experiment</td><td class="tg-0lax">Date/Time</td>    <td class="tg-0lax">ID</td>    <td class="tg-0lax">Run</td>    <td class="tg-0lax">Pretraining</td>    <td class="tg-0lax">Validation</td>    <td class="tg-0lax">Fischer Coefficients</td>    <td class="tg-0lax">Pretrain ABX</td>    <td class="tg-0lax">Training</td>    <td class="tg-0lax">Results Processing</td>    <td class="tg-0lax">Train ABX</td>  </tr>'
    html_suffix = '</tbody>\n</table>'

    full_html = html_prefix + html_firstrow+table_str + html_suffix

    f = open('loghtml/logtable_'+experiment_folder + '.html',  'w+')
    f.write(full_html)
    f.close()


