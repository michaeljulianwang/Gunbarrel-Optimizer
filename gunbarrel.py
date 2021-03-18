import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize, differential_evolution, basinhopping, brute
from random import random as rd
import datetime

class GunbarrelOptimizer:

    def __init__(self, num_zones, well_counts, depths, existing_wells, lease_offset, right_bound):

        self.num_zones = num_zones
        self.well_counts = well_counts
        self.depths = list(sorted(depths))
        self.existing_wells = existing_wells
        self.lease_offset = lease_offset
        self.right_bound = right_bound

        self.validateInputs()

    def validateInputs(self):

        assert self.num_zones == len(self.well_counts) == len(self.depths)
        assert max(self.depths) < 0

        return

    def leaseOffsetConstraint(self, x): # min lease offset of X from both sides

        minx = min(x) - 0
        maxx = self.right_bound - max(x)

        return min(minx, maxx) - self.lease_offset

    def sumCost(self, zoneA, zoneB):
        ''' Computes minimum distance between combinations of wells across 2 zones; compares within and between zones '''

        z = distance.cdist(
            XA=zoneA,
            XB=zoneB
        ) # check distances between zones

        if len(zoneA) == len(zoneB) == 1:
            pass # the distance between the 2 points is already z
        elif len(zoneA) == 1:
            z = min(z.min(), np.diff(zoneB[:, 0]).min())
        elif len(zoneB) == 1:
            z = min(z.min(), np.diff(zoneA[:, 0]).min())
        else:
            z = min(z.min(), np.diff(zoneA[:, 0]).min(), np.diff(zoneB[:, 0]).min())

        return z

    def objFun(self, x):
        ''' Returns the smallest (closest) distance between wells across all zones, as negative number. '''

        list_of_arrays = []
        x_cumsum = np.cumsum(self.well_counts)
        for i in range(len(self.well_counts)):
            if i == 0:
                x_zone = x[0:x_cumsum[i]]
            else:
                x_zone = x[x_cumsum[i - 1]:x_cumsum[i]]
            x_zone_array = np.array([(ele, self.depths[i]) for ele in x_zone])
            list_of_arrays.append(x_zone_array)

        s = np.inf
        for i in range(len(list_of_arrays) - 1):
            s = min(s, self.sumCost(list_of_arrays[i], list_of_arrays[i + 1])) # compare with 1 zone ahead
            try:
                s = min(s, self.sumCost(list_of_arrays[i], list_of_arrays[i + 2])) # compare with 2 zones ahead
            except:
                pass

        if len(self.existing_wells) > 0:
            for i in range(len(list_of_arrays)): # compare each zone with each existing wells
                for j in range(len(self.existing_wells)):
                    s = min(s, self.sumCost(list_of_arrays[i], self.existing_wells[j, :].reshape(1, 2)))
        return -s

    def visualize(self, x):
        ''' Plot existing wells and proposals '''

        list_of_arrays = []
        x_cumsum = np.cumsum(self.well_counts)
        for i in range(len(self.well_counts)):
            if i == 0:  # first element
                x_zone = x[0:x_cumsum[i]]
            else:
                x_zone = x[x_cumsum[i - 1]:x_cumsum[i]]
            x_zone_array = np.array([(ele, self.depths[i]) for ele in x_zone])
            list_of_arrays.append(x_zone_array)

        for i in range(len(list_of_arrays)):
            plt.plot(list_of_arrays[i][:, 0], list_of_arrays[i][:, 1], '2', color='navy', label='Proposal' if i == 0 else '')

        if len(self.existing_wells) > 0:
            plt.plot(self.existing_wells[:, 0], self.existing_wells[:, 1], '1', color='red', label='Existing')

        plt.axis('equal')
        plt.title('Closest 2D distance between 2 wells: ' + '%.3f' % -self.objFun(x) + ' ft')
        plt.xlabel('HZ offset')
        plt.ylabel('Depth')
        plt.yticks(rotation=45)
        plt.legend()
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 1
        plt.rcParams['grid.color'] = "#cccccc"
        plt.axvline(x=0, color='r')
        plt.axvline(x=self.right_bound, color='r')
        plt.grid(True)
        plt.show()

        return

    def initializeGuesses(self):
        ''' Instantiate initial x0 values for scipy.minimize and scipy.basinhopping to iterate '''

        x = []
        for i in range(self.num_zones):
            wells_in_zone = self.well_counts[i]
            initial_spacing = self.right_bound / wells_in_zone
            well_iter = 0
            if i % 2 == 0:
                while self.lease_offset + well_iter * initial_spacing <= self.right_bound:
                    x.append(self.lease_offset + well_iter * initial_spacing + 2 * (rd() - 0.5))
                    well_iter += 1
            else:
                while (self.right_bound - self.lease_offset - (wells_in_zone - 1) * initial_spacing) + well_iter * initial_spacing + 2 * (rd() - 0.5) <= self.right_bound:
                    x.append((self.right_bound - self.lease_offset - (wells_in_zone - 1) * initial_spacing)+ well_iter * initial_spacing + 2 * (rd() - 0.5))
                    well_iter += 1
        if len(x) != sum(self.well_counts):
            return -1
        else:
            return x

    def localMinimize(self, x0_=None, plotOn=True):
        ''' Local minimization. Can get stuck in local optimas. '''

        if x0_ is None:
            x0_ = self.initializeGuesses()

        cons = ({'type': 'ineq', 'fun': self.leaseOffsetConstraint})  # constraints
        a = minimize(fun=self.objFun, x0=x0_, method='SLSQP', constraints=cons, tol=10 ** -10, options={'maxiter': 1000})

        if plotOn:
            self.visualize(a.x)

        return a

    def differentialEvo(self, iters=10, plotOn=True):
        '''
        Differential Evolution

        Notes:
        - smaller length of bounds --> faster iterations
        - popsize: higher --> slower iterations but will not get stuck in local optimas
        - strategy (best to worst): randtobest1exp, currenttobest1bin, best1bin, randtobest1bin
        - smaller bounds increases probability of reaching global optimum + faster run times
        '''

        start = datetime.datetime.now()

        bounds_ = [(self.lease_offset, self.right_bound-self.lease_offset) for _ in range(sum(self.well_counts))]
        # bounds_ = [
        #     (330, 1100), (1100, 1870), (1870, 2640), (2640, 3410), (3410, 4180), (4180, 4950),
        #     (330, 1100), (1100, 1870), (1870, 2640), (2640, 3410), (3410, 4180), (4180, 4950),
        #     (330, 1100), (1100, 1870), (1870, 2640), (2640, 3410), (3410, 4180), (4180, 4950),
        #     (330, 1100), (1100, 1870), (1870, 2640), (2640, 3410), (3410, 4180), (4180, 4950),
        #     (330, 1100), (1100, 1870), (1870, 2640), (2640, 3410), (3410, 4180), (4180, 4950),
        # ]

        best_s = np.inf
        best_a = None

        print('Executing ' + str(iters) + ' iterations to find optimal spacing...')

        for i in range(iters):
            a = differential_evolution(func=self.objFun, bounds=bounds_,strategy='randtobest1exp', maxiter=1000,
                                       tol=10 ** -5, disp=False, popsize=40,recombination=1, workers=10, mutation=0.8)
            a = self.localMinimize(x0_=a.x, plotOn=False) # pass thru local minimizer

            if a.fun < best_s:
                best_s = a.fun
                best_a = a

            print('Iteration ' + str(i + 1) + ' of ' + str(iters) + ' - ' + 'Closest 2D distance between 2 wells: ' + str(best_s))

        end = datetime.datetime.now()
        print('Time taken to reach solution: ' + str(end - start))

        if plotOn:
            self.visualize(best_a.x)

        return a

    def basinHop(self, plotOn=True):
        ''' Basin hopping '''

        bounds_ = [(self.lease_offset, self.right_bound-self.lease_offset)] * sum(self.well_counts)
        x0_ = self.initializeGuesses()
        a = basinhopping(func=self.objFun, x0=x0_, niter=1000,minimizer_kwargs={'bounds': bounds_, 'method': 'SLSQP'},
                         disp=True, stepsize=5000)

        a = self.localMinimize(x0_=a.x, plotOn=False) # pass thru local minimizer

        if plotOn:
            self.visualize(a.x)

        return a

    def brute(self, plotOn=True):
        ''' Brute force. Recommended for small zones/heavily bounded problems '''

        # rranges = tuple([slice(self.lease_offset, self.right_bound-self.lease_offset, 100) for _ in range(sum(self.well_counts))])
        rranges = tuple([(self.lease_offset, self.right_bound-self.lease_offset) for _ in range(sum(self.well_counts))])

        bruteX = brute(func=self.objFun, ranges=rranges, disp=True, full_output=False, finish=None, workers=-1)
        a = self.localMinimize(x0_=bruteX) # pass thru local minimizer

        if plotOn:
            self.visualize(a.x)

        return a

    def getCoords(self, x):
        ''' Get (x, y) coordinates of proposed well locations '''

        # collect depths
        proposal_depths = []
        for zone in range(self.num_zones):
            zone_well_count = self.well_counts[zone]
            zone_depth = self.depths[zone]
            proposal_depths.extend([zone_depth] * zone_well_count)

        assert len(x) == len(proposal_depths)

        # pair depths with proposed x's
        coords = []
        for i in range(len(x)):
            coords.append((x[i], proposal_depths[i]))

        return coords

if __name__ == "__main__":

    ### INPUTS ###

    num_zones_ = 3 # num of zones to add
    well_counts_ = [3, 2, 3] # num of wells per zone to add; length == `num_zones_`
    depths_ = [-10000, -9000, -8000] # depth of each zone; length == `num_zones_`
    existing_wells_ = np.array([
        (330, -8000),
        (2640, -10000)
    ]) # (x, y) of existing wells to respect
    lease_offset_ = 330 # distance to maintain from 0 and `right_bound_`
    right_bound_ = 5000 # right-bound of lease. left-bound of lease defaults to 0

    ### INPUTS ###

    go = GunbarrelOptimizer(
        num_zones=num_zones_,
        well_counts=well_counts_,
        depths=depths_,
        existing_wells=existing_wells_,
        lease_offset=lease_offset_,
        right_bound=right_bound_
    )

    # Differential Evolution
    de = go.differentialEvo()
    go.getCoords(de.x)

    # Basin Hopping
    bh = go.basinHop()
    go.getCoords(bh.x)

    # Local Minimizer
    lm = go.localMinimize()
    go.getCoords(lm.x)

    # Brute Force
    b = go.brute()
    go.getCoords(b.x)
