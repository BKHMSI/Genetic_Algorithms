import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for GA')
    # Hyper-Parameters
    parser.add_argument('--g',  type=int,   default=100,  help='number of generations')
    parser.add_argument('--n',  type=int,   default=8,    help='string length')
    parser.add_argument('--N',  type=int,   default=10,   help='population size')
    parser.add_argument('--pc', type=float, default=0.6,  help='crossover probability')
    parser.add_argument('--pm', type=float, default=1/8., help='mutation probability')
    return parser.parse_args()

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def average(population):
    avg = 0.
    for p in population:
        avg += fitness(p)
    return avg/len(population)

def maximum(population):
    m = fitness(population[0])
    for p in population[1:]:
        score = fitness(p)
        m = score if score > m else m
    return m

def minimum(population):
    m = fitness(population[0])
    for p in population[1:]:
        score = fitness(p)
        m = score if score < m else m
    return m

def stats(population):
    print("-"*10)
    print("avg: {}".format(average(population)))
    print("max: {}".format(maximum(population)))
    print("min: {}".format(minimum(population)))
    print("-"*10)

def initialze(N, n):
    ''' random initialization '''
    return np.random.randint(2, size=(N,n))

def selection(population, N):
    ''' binary tournament selection '''
    indecies = np.random.randint(N, size=(N,2))
    tournament = np.array(population)[indecies]
    winners = [((t[0]) if fitness(t[0]) > fitness(t[1]) else (t[1])) for t in tournament]
    return winners

def variation(parents, args):
    ''' variation to create new solutions through crossover and mutation '''
    return mutation(crossover(parents, args), args)

def crossover(parents, args):
    ''' one-point crossover '''
    cross = []
    probs = np.random.choice([True,False], p=[args.pc, 1-args.pc], size=args.N/2)
    for i, parent in enumerate(grouped(parents,2)):
        if probs[i]:
            point = np.random.randint(args.n)
            temp = parent[0][point:]
            parent[0][point:] = parent[1][point:]
            parent[1][point:] = temp 
        cross.append(parent[0])
        cross.append(parent[1])
    return cross

def mutation(parents, args):
    ''' bit-flip mutation '''
    assert(args.pm == 1./args.n)
    mutate = []
    for parent in parents:
        mask   = np.random.choice([1,0], p=[args.pm, 1-args.pm], size=args.n)
        parent = parent ^ mask
        mutate.append(list(parent)) 
    return mutate

def replacement(children):
    ''' full replacement '''
    return children

def fitness(chromosome):
    ''' fitness function '''
    return np.array(chromosome).sum()

def main():
    args = parse_args()
    population = initialze(args.N, args.n)
    stats(population)
    for g in range(args.g):
        parents = selection(population, args.N)
        children = variation(parents, args)
        population = replacement(children)
    stats(population)

if __name__ == "__main__":
    main()