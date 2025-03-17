import random
from dataclasses import dataclass
import embedding as em
import numpy as np
import utils as ut
from termcolor import colored

class AbcSol:
    instance_count = 0
    def __init__(self, source_number, t, block_size, count=True):
        AbcSol.instance_count += 1
        self.id = AbcSol.instance_count
        self.source = source_number
        self.trail = 0
        self.error = 0
        self.prediction = 0
        self.abandoned = False

        self.t = t
        self.block_size = block_size

    def __repr__(self):
        s = ''
        # s = f'{'sol ' + str(self.id):<7} '
        if self.source != -1:
            s = s + f'source({self.source}) '
        s = s + f'trail{'(' + str(self.trail) + ')':<6}'
        s = s + f'error({self.error:.15f}) '
        s = s + f'pred({self.prediction:.6f})   '
        s = s + ('x' if self.abandoned else ' ')
        s = s + f'   t({self.t:.6f}) '
        s = s + f'b_size({self.block_size})'
        return s

class AbcSpace:
    def __init__(self):
        self.t = (1e-5, 1)
        self.block_size = (4, 4)
    
    def generate_sol(self, source_number=None):
        return AbcSol(
            source_number=(source_number if source_number is not None else -1),
            t=max(random.uniform(self.t[0], self.t[1]), 0.000001),
            block_size=random.randint(self.block_size[0], self.block_size[1])
        )

class AbcParams:
    def __init__(self, process_params=em.ProcessParams()):
        self.food_number = 5
        self.trail_limit = 50
        self.iteration_limit = 100
        self.debug = False
        self.process_params = process_params
        self.show_food_sources = False
        self.show_images_at_end = True


class Abc:
    def __init__(self, img, mess, space=AbcSpace(), params=AbcParams()):
        self.space = space
        self.params = params
        self.img = img
        self.mess = mess
    
    def run(self):
        # init food sources
        food_sources = [[] for _ in range(self.params.food_number)]

        if not self.params.debug:
            ut.printProgressBar(0, self.params.iteration_limit + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

        # init 0 generation
        for i in range(self.params.food_number):
            food_sources[i].append(self.space.generate_sol(i + 1))
        
        for i in range(self.params.food_number):
            self.process_img(food_sources[i][-1])
        
        self.fill_predictions(food_sources)

        best_sol = self.get_best(food_sources)

        if not self.params.debug:
            ut.printProgressBar(1, self.params.iteration_limit + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

        
        for i in range(self.params.iteration_limit):
            # init new generation
            for n, source in enumerate(food_sources):
                last = source[-1]
                new = self.gen_solution(source, n + 1)
                self.process_img(new)

                if self.params.debug:
                    print('in new generation')
                    print(last)
                    print(new)
                    print(f'got: {ut.pbool(new.error < last.error)}\n')

                if new.error < last.error:
                    source.append(new)
                else:
                    last.trail += 1
            self.fill_predictions(food_sources)

            # check for random limit
            for n, source in enumerate(food_sources):
                limit = random.uniform(0, 1)
                if source[-1].prediction > limit:
                    last = source[-1]
                    new = self.gen_solution(source, n + 1)
                    self.process_img(new)

                    if self.params.debug:
                        print('in limit generation')
                        print(last)
                        print(new)
                        print(f'got: {ut.pbool(new.error < last.error)}\n')

                    if new.error < last.error:
                        source.append(new)
                    else:
                        last.trail += 1
            self.fill_predictions(food_sources)

            # update best solution
            local_best = self.get_best(food_sources)
            if local_best.error <= best_sol.error:
                best_sol = local_best
            
            # update abandoned 
            for n, source in enumerate(food_sources):
                if source[-1].trail >= self.params.trail_limit:
                    source[-1].abandoned = True
                    new = self.space.generate_sol(n + 1)
                    self.process_img(new)
                    source.append(new)
            
            self.fill_predictions(food_sources)

            # update best solution
            local_best = self.get_best(food_sources)
            if local_best.error <= best_sol.error:
                best_sol = local_best
            
            if not self.params.debug:
                ut.printProgressBar(i + 2, self.params.iteration_limit + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        if self.params.show_food_sources:
            self.print_sources(food_sources, best_sol)
        
        print(f'{colored('best', 'green')}: {best_sol}\n')

        if self.params.show_images_at_end:
            process_params = em.ProcessParams()
            process_params.show = True
            stats = self.process_img(best_sol, process_params=process_params)
        else:
            stats = self.process_img(best_sol)
        
        print(stats)
        return best_sol
    
    def run_random(self):
        sol = self.space.generate_sol()
        process_params = em.ProcessParams()
        process_params.debug = self.params.debug
        process_params.show = self.params.show_images_at_end
        stats = self.process_img(sol, process_params)
        print(stats)
        return sol

    def get_best(self, food_sources: list[AbcSol]) -> AbcSol:
        best = food_sources[0][-1]
        for source in food_sources:
            if source[-1].error < best.error:
                best = source[-1]
        return best

    def gen_solution(self, source: list[AbcSol], source_number):
        curr = source[-1]
        
        mf = random.uniform(-1, 1)

        if self.params.debug:
            print(f'mf: {mf:.10f}')
        
        pos = len(source) - 1
        while pos > 0 and not source[pos - 1].abandoned:
            pos -= 1
        
        if pos == (len(source) - 1):
            new_t = max(curr.t + mf * curr.t, 0.000001)
            new = AbcSol(source_number, new_t, curr.block_size)
        else:
            past = source[random.randint(pos, len(source) - 2)]
            new_t = max(curr.t + mf * (curr.t - past.t), 0.000001)
            new = AbcSol(source_number, new_t, curr.block_size)
        
        return new
        

    def print_sources(self, food_sources, best=None):
        for i, food in enumerate(food_sources):
            print(f'food {i + 1}: ')
            for n, sol in enumerate(food):
                num = f'{str(n+1) + '.':<4}'
                sol_str = str(sol)
                if best is not None and sol == best:
                    num = colored(num, 'green')
                    sol_str = colored(sol_str, 'green')
                print(f'   {num} ', end='')
                print(sol_str)
    
    def fill_predictions(self, food_sources):
        error_sum = sum(food[-1].error for food in food_sources)
        # if self.params.debug:
        #     print(f'error_sum = {error_sum}')

        for sol in food_sources[:][-1]:
            if sol.error >= 0:
                prediction = 1 / (sol.error + 1)
            else:
                prediction = 1 + abs(sol.prediction)
            sol.prediction = prediction
    
    def process_img(self, sol, process_params=None):
        if process_params is None:
            process_params=self.params.process_params
        stats = em.process(self.img, self.mess, sol.t, sol.block_size, process_params)
        sol.error = stats.error_function
        return stats