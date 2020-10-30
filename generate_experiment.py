import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("root", help="root directory")
parser.add_argument("state", help="state file to use as input")
parser.add_argument("out", help="output file")
parser.add_argument("episode_length", help="length of each episode")
parser.add_argument("num_episodes", help = "total number of episodes")
parser.add_argument("-consistent", help= "should experiments be consistent?  ie. same states in each episode")
parser.add_argument("-overwrite", help="overwrite any existing output files")
args = parser.parse_args()




with open(args.root+ '/' + args.state) as f:
    input_data = f.read().splitlines()
    header = input_data[0].split('\t')
    assert header[0] == 'state'

if args.overwrite:
    write_method = 'w'
else:
    write_method = 'x'
outfile = open(args.root + args.out , write_method)
outfile.close()

states = []

for line in input_data[1:]:
    assert line.split('\t')[0] not in states
    states.append(line.split('\t')[0])

print('loaded '+str(len(states))+' states')
episodes = []
if args.consistent:
    print('consistent')
    assert int(args.episode_length) % len(states) == 0
    reps = int(args.episode_length) / len(states)

    for x in range(int(args.num_episodes)):
        ep = ['eps'+str(x)]
        for y in range(int(reps)):
            new_list = states.copy()
            random.shuffle(new_list)
            ep = ep + new_list
        episodes.append(ep)
else:
    for x in range(int(args.num_episodes)):
        ep = ['eps'+str(x)]
        for y in range(int(args.episode_length)):
            ep.append(random.choice(states))


outfile = open(args.root+'/' +args.out, 'a+')
for e in episodes:
    outfile.write(''.join([i+' ' for i in e[:-1]])+ e[-1])
    outfile.write('\n')

outfile.close()
