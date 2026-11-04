import numpy as np
import copy
from lib.vrp_helper_1 import get_solDist
import math

def build_edge_index_routes(batch_routes, inverse=False):
    edge_index_routes = []
    for i_batch in range(len(batch_routes)):
        froms = []
        tos = []
        routes = batch_routes[i_batch]
        for r in routes:
            r0 = [0]+r+[0]
            if inverse:
                for i_tmp in range(len(r0)-2):
                    tos.append(r0[i_tmp])
                    froms.append(r0[i_tmp+1])
            else:
                for i_tmp in range(1,len(r0)-1):
                    froms.append(r0[i_tmp])
                    tos.append(r0[i_tmp+1])
        edge_index_routes.append(np.array([froms, tos]))
    return edge_index_routes

def build_edge_index_near(distMats, n, inverse=False): # n nombre de neighbors
    edge_index_routes = []
    for i_batch in range(len(distMats)):
        froms = []
        tos = []
        distMat = distMats[i_batch]
        for anchor in range(len(distMat)):
            for node in np.argsort(distMat[anchor])[1:n+1]:
                if inverse:
                    tos.append(node)
                    froms.append(anchor)
                else:
                    froms.append(node)
                    tos.append(anchor)
        edge_index_routes.append(np.array([froms, tos]))
    return edge_index_routes

def build_action_index(n_nodes):
    result = []
    for i in range(1,n_nodes+1):
        for j in range(1,n_nodes+1):
            result.append((i,j,True))
    for i in range(1,n_nodes+1):
        for j in range(1,n_nodes+1):
            result.append((i,j,False))
    return result

def find_ruins(n_ruins, anchor, coefficient, neighbour, routes, ruined):
    """
    与 SISR 规则类似。区别在于:
        1. 路线中被删去部分的比例是输入项，而非通过 l_max, c_bar 等参数随机。
        2. 如果下一个 neighbour 所在的线路被删除过，该点依然可以被删除，且一样会寻找线路前后的点删除。
    以上修改使得 coefficient 和删去点的空间聚集度具有更严格的正相关性
    
    Similaire aux règles SISR. La différence est que :
         1. La proportion de la partie supprimée dans l'itinéraire est un élément d'entrée, non aléatoire via des paramètres tels que l_max et c_bar.
         2. Si la ligne où se trouve le prochain voisin a été supprimée, le point peut toujours être supprimé et les points avant et après la ligne seront également recherchés pour suppression.
     Les modifications ci-dessus confèrent au coefficient une corrélation positive plus stricte avec l'agrégation spatiale des points supprimés.
    """
    
    """
    So cette fonction :
    prend en entrée : un anchor node et son coefficient de route et le degrée de destruction par anchor_node n_ruins (ici c'est )
    Output : 
        pour chaque c voisin de anchor node; et si c n'est pas déja dans ruin_nodes; et c n'est pas le dépot :
            1) on calcule le nombre de noeuds à supprimer int(np.ceil(len(routes[t]) * coefficient)) avec t la route de c
            2) on prend 1 à gauche et 1 à doite du anchor_node jusqu'à atteindre le nombre de noeuds à supprimer pour ce neighbor
            3) stop at n_ruins
            
    """
    def find_t(routes, c):
        for i in range(len(routes)):
            if c in routes[i]: return i
        return None
    
    def remove_nodes(route, c, num):
        nodes = []
        i = route.index(c)
        nodes.append(route[i])
        j = i+1
        k = i-1
        num = min(num, len(route))
        while len(nodes)<num:
            if j>=len(route):
                j=0 
            if k<0:
                k = len(route)-1
            nodes.append(route[j])
            if len(nodes)<num:
                nodes.append(route[k])
            j+=1
            k-=1
        return nodes

    coefficient = max(coefficient, 1e-8)
    ruin_nodes = []
    # ruined_t_indices = set([])
    for c in neighbour[anchor]:
        if c not in ruin_nodes and c!=0:
            #print(f"\n avant find_t routes = {routes}, c = {c}")
            t = find_t(routes, c)
            #print(f"\n indice t = {t}")
            # if t in ruined_t_indices: continue
            num = int(np.ceil(len(routes[t]) * coefficient))
            newly_removed = remove_nodes(routes[t], c, num)
            for node in newly_removed:
                if node not in ruin_nodes and node not in ruined and len(ruin_nodes)<n_ruins: ruin_nodes.append(node)
            # ruined_t_indices.add(t)
            if len(ruin_nodes)>= n_ruins:
                break
    ruin_nodes = ruin_nodes[:n_ruins]  
    #     print(ruined, ruin_nodes)
    return ruin_nodes

class vrp_env:
    
    def __init__(self, cap, data, distMat, init_routes, init_dist, init_T, alpha_T, sim_anl=True):
        self.cap = cap
        self.data = data
        self.distMat = distMat
        self.curr_routes = init_routes
        self.curr_dist = init_dist
        self.best_routes = copy.deepcopy(init_routes)
        self.best_dist = init_dist
        self.T = init_T
        self.alpha_T = alpha_T
        self.sim_anl = sim_anl
        
    def get_embedding(self):
        result = []
        norm = self.data[0,4]-self.data[0,3]
        result.append(self.data[:,0]/norm) # x normalized by depot's 时间窗跨度
        result.append(self.data[:,1]/norm) # y normalized by depot's 时间窗跨度
        result.append(self.data[:,2]/self.cap)
        result.append(self.data[:,3]/norm) # start_time normalized by depot's 时间窗跨度
        result.append(self.data[:,4]/norm) # end_time normalized by depot's 时间窗跨度
        result.append(self.data[:,5]/norm) # service_time normalized by depot's 时间窗跨度
        result.append((self.data[:,4]-self.data[:,3])/norm) # 时间窗跨度除以 depot's 时间窗跨度
        result.append(np.zeros(len(self.data))) # cumulative demand normalized by cap
        result.append(np.zeros(len(self.data))) # cumulative distance normalized by depot's 时间窗跨度
        result.append(np.zeros(len(self.data))) # total demand normalized by cap
        result.append(np.zeros(len(self.data))) # total distance normalized by depot's 时间窗跨度
        for r in self.curr_routes:
            complete_r = [0]+r+[0]
            cum_demand = 0
            cum_dist = 0
            for i in range(1, len(complete_r)-1):
                cum_demand+=self.data[complete_r[i],2]
                cum_dist+=self.distMat[complete_r[i-1],complete_r[i]]
                result[7][complete_r[i]] = cum_demand
                result[8][complete_r[i]] = cum_dist
            for i in range(1, len(complete_r)-1):
                result[9][complete_r[i]] = cum_demand
                result[10][complete_r[i]] = cum_dist

        result[7]/=self.cap
        result[8]/=norm
        result[9]/=self.cap
        result[10]/=norm
        return np.transpose(np.array(result))
    
    def reward_func(self, old_d, new_d):
        # return max(-1, old_d-new_d)
        return old_d-new_d
    
    def sim_annealing(self, new_dist, new_routes):
        if new_dist<(self.curr_dist-self.T*np.log(np.random.random())):
            if new_dist<self.best_dist:
                self.best_dist = new_dist
                self.best_routes = copy.deepcopy(new_routes)
            self.curr_dist = new_dist
            self.curr_routes = new_routes

    def step(self, action):
        def find_ij(routes, node):
            for i,r in enumerate(routes):
                for j,c in enumerate(r):
                    if c==node:
                        return i,j
            return -1,-1

        def check(vehicle_capcity, data, distance_matrix, r):
            R = [0]+r+[0]
            t = 0
            demands = np.sum([data[x,2] for x in R])
            if demands>vehicle_capcity: return False
            for i in range(len(R)):
                if i==0:
                    arrive_t = t
                else:
                    arrive_t = t+distance_matrix[R[i-1],R[i]]
                start_t = max(data[R[i],3], arrive_t)
                due_t = data[R[i],4]
                serve_t = data[R[i],5]
                end_t = start_t+serve_t
                t=end_t
                if due_t<start_t: return False
            return True

        def get_route_distance(distance_matrix, route):
            r = [0]+route+[0]
            result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
            return result

        node1, node2, isPrevious = action
        ij1 = find_ij(self.curr_routes, node1)
        ij2 = find_ij(self.curr_routes, node2)
        if ij1[0]!=ij2[0]:
            r2 = copy.deepcopy(self.curr_routes[ij2[0]])+[0]
            if isPrevious:
                r2 = r2[:ij2[1]]+[node1]+r2[ij2[1]:]
            else:
                r2 = r2[:(ij2[1]+1)]+[node1]+r2[(ij2[1]+1):]
            r2 = r2[:-1]
            if check(self.cap, self.data, self.distMat, r2):
                new_routes = copy.deepcopy(self.curr_routes)
                new_routes[ij1[0]] = new_routes[ij1[0]][:ij1[1]]+new_routes[ij1[0]][(ij1[1]+1):]
                new_routes[ij2[0]] = r2
                dist = get_solDist(self.distMat, new_routes)
                r = self.reward_func(self.curr_dist, dist)
                if self.sim_anl:
                    self.sim_annealing(dist, new_routes)
                else:
                    self.curr_routes = new_routes
                    self.curr_dist = dist
                self.T *= self.alpha_T
            else:
                r = -1.0
        else:
            r1 = copy.deepcopy(self.curr_routes[ij2[0]])+[0]
            r1 = r1[:ij1[1]]+r1[(ij1[1]+1):]
            try:
                j2 = r1.index(node2)
            except ValueError:
                print(r1, action, self.curr_routes)
                raise
            if isPrevious:
                r1 = r1[:j2]+[node1]+r1[j2:]
            else:
                r1 = r1[:(j2+1)]+[node1]+r1[(j2+1):]
            r1 = r1[:-1]
            if check(self.cap, self.data, self.distMat, r1):
                new_routes = copy.deepcopy(self.curr_routes)
                new_routes[ij1[0]] = r1
                dist = get_solDist(self.distMat, new_routes)
                r = self.reward_func(self.curr_dist, dist)
                if self.sim_anl:
                    self.sim_annealing(dist, new_routes)
                else:
                    self.curr_routes = new_routes
                    self.curr_dist = dist
                self.T *= self.alpha_T
            else:
                r = -1.0

        return r
    
class vrp_env_sisr:
    
    def __init__(self, cap, data, distMat, init_routes, init_dist, init_T, alpha_T, sim_anl=True):
        self.cap = cap
        self.data = data
        self.distMat = distMat
        self.curr_routes = init_routes
        self.curr_dist = init_dist
        self.best_routes = copy.deepcopy(init_routes)
        self.best_dist = init_dist
        self.neighbours = self.get_neighbours()
        self.T = init_T
        self.alpha_T = alpha_T
        self.sim_anl = sim_anl
        
    def get_embedding(self):
        result = []
        norm = self.data[0,4]-self.data[0,3]
        result.append(self.data[:,0]/norm) # x normalized by depot's 时间窗跨度                         (x coordinate) # normalized by depot's time window span
        result.append(self.data[:,1]/norm) # y normalized by depot's 时间窗跨度                         (y coordinate)
        result.append(self.data[:,2]/self.cap)   #                                                     (demand)
        result.append(self.data[:,3]/norm) # start_time normalized by depot's 时间窗跨度                (starting time of the service)
        result.append(self.data[:,4]/norm) # end_time normalized by depot's 时间窗跨度                  (ending time of the service)
        result.append(self.data[:,5]/norm) # service_time normalized by depot's 时间窗跨度              (service time period)
        result.append((self.data[:,4]-self.data[:,3])/norm) # 时间窗跨度除以 depot's 时间窗跨度          (Time window span divided by depot's time window span)
        result.append(np.zeros(len(self.data))) # cumulative demand normalized by cap                  (sum up of demands qi of the corresponding route till this node)
        result.append(np.zeros(len(self.data))) # cumulative distance normalized by depot's 时间窗跨度  (accumulated traveling distance along this route till this node)
        result.append(np.zeros(len(self.data))) # total demand normalized by cap                       (a total demand of all nodes on the corresponding route)
        result.append(np.zeros(len(self.data))) # total distance normalized by depot's 时间窗跨度       (a total traveling distance of the corresponding route)
        for r in self.curr_routes:
            complete_r = [0]+r+[0]
            cum_demand = 0
            cum_dist = 0
            for i in range(1, len(complete_r)-1):
                cum_demand+=self.data[complete_r[i],2]
                cum_dist+=self.distMat[complete_r[i-1],complete_r[i]]
                result[7][complete_r[i]] = cum_demand
                result[8][complete_r[i]] = cum_dist
            for i in range(1, len(complete_r)-1):
                result[9][complete_r[i]] = cum_demand
                result[10][complete_r[i]] = cum_dist

        result[7]/=self.cap
        result[8]/=norm
        result[9]/=self.cap
        result[10]/=norm
        return np.transpose(np.array(result))
    
    
    
    def get_embedding_cvrp(self):
        result = []

        min_val_x = np.min(self.data[:,0])
        max_val_x = np.max(self.data[:,0])
        
        min_val_y = np.min(self.data[:,1])
        max_val_y = np.max(self.data[:,1])
        
        #norm =  abs(self.data[:,1] - self.data[:,0] )       
        result.append((self.data[:,0] - min_val_x) / (max_val_x - min_val_x)) # x normalized           0(x coordinate) 
        #result.append(self.data[:,0] / norm) # x normalized           0(x coordinate) 
        result.append((self.data[:,1] - min_val_y) / (max_val_y - min_val_y)) # y normalized           1(y coordinate)
        #result.append(self.data[:,1] / norm) # y normalized           1(y coordinate)
        result.append(self.data[:,2]/self.cap)   #                                                     2(demand)
        #result.append(self.data[:,3]/norm) # start_time normalized by depot's 时间窗跨度                (starting time of the service)
        #result.append(self.data[:,4]/norm) # end_time normalized by depot's 时间窗跨度                  (ending time of the service)
        #result.append(self.data[:,5]/norm) # service_time normalized by depot's 时间窗跨度              (service time period)
        #result.append((self.data[:,4]-self.data[:,3])/norm) # 时间窗跨度除以 depot's 时间窗跨度          (La durée de la fenêtre de temps divisée par la durée de la fenêtre de temps du dépôt)
        result.append(np.zeros(len(self.data))) # cumulative demand normalized by cap                  3(sum up of demands qi of the corresponding route till this node)
        result.append(np.zeros(len(self.data))) # cumulative distance normalized by depot's 时间窗跨度  4(accumulated traveling distance along this route till this node)
        result.append(np.zeros(len(self.data))) # total demand normalized by cap                       5(a total demand of all nodes on the corresponding route)
        result.append(np.zeros(len(self.data))) # total distance normalized by depot's 时间窗跨度       6(a total traveling distance of the corresponding route)
        
        for r in self.curr_routes:
            complete_r = [0]+r+[0]
            cum_demand = 0
            cum_dist = 0
            for i in range(1, len(complete_r)-1):
                cum_demand+=self.data[complete_r[i],2]
                cum_dist+=self.distMat[complete_r[i-1],complete_r[i]]
                result[3][complete_r[i]] = cum_demand
                result[4][complete_r[i]] = cum_dist
            for i in range(1, len(complete_r)-1):
                result[5][complete_r[i]] = cum_demand
                result[6][complete_r[i]] = cum_dist

        result[3]/=self.cap
        #result[4]/=norm
        result[5]/=self.cap
        #result[6]/=norm
        return np.transpose(np.array(result))
    
    
    
    
    def get_embedding_cvrp_metrics(self):
        result = []

        min_val_x = np.min(self.data[:,0])
        max_val_x = np.max(self.data[:,0])
        
        min_val_y = np.min(self.data[:,1])
        max_val_y = np.max(self.data[:,1])
        
        result.append((self.data[:,0] - min_val_x) / (max_val_x - min_val_x)) # x normalized           0(x coordinate) 
        result.append((self.data[:,1] - min_val_y) / (max_val_y - min_val_y)) # y normalized           1(y coordinate)
        result.append(self.data[:,2]/self.cap)  #                                                      2(demand)
        
        result.append(np.zeros(len(self.data))) # cumulative demand normalized by cap                  3(sum up of demands qi of the corresponding route till this node)
        result.append(np.zeros(len(self.data))) # cumulative distance normalized by depot's 时间窗跨度  4(accumulated traveling distance along this route till this node)
        result.append(np.zeros(len(self.data))) # total demand normalized by cap                       5(a total demand of all nodes on the corresponding route)
        result.append(np.zeros(len(self.data))) # total distance normalized by depot's 时间窗跨度       6(a total traveling distance of the corresponding route)
        
        result.append(np.zeros(len(self.data))) # average width per route  (S5)                        7 
        result.append(np.zeros(len(self.data))) # average span in radian per route   (S6)              8 
        result.append(np.zeros(len(self.data))) # average compactness per route mesured by width (S7)  9 
        result.append(np.zeros(len(self.data))) # average compactness per route mesured by radian (S8) 10 
        
        
        
        # calculate cumulative demand and distance and total demand and distance
        for r in self.curr_routes:
            complete_r = [0]+r+[0]
            cum_demand = 0
            cum_dist = 0
            for i in range(1, len(complete_r)-1):
                cum_demand+=self.data[complete_r[i],2]
                cum_dist+=self.distMat[complete_r[i-1],complete_r[i]]
                result[3][complete_r[i]] = cum_demand
                result[4][complete_r[i]] = cum_dist
            for i in range(1, len(complete_r)-1):
                result[5][complete_r[i]] = cum_demand
                result[6][complete_r[i]] = cum_dist

        # Normalize
        result[3]/=self.cap
        result[5]/=self.cap
        
        min_cumul_distance = np.min(result[4])
        max_cumul_distance = np.max(result[4])
        result[4] = (result[4] - min_cumul_distance) / (max_cumul_distance - min_cumul_distance)
        
        min_total_distance = np.min(result[6])
        max_total_distance = np.max(result[6])
        result[6] = (result[6] - min_total_distance) / (max_total_distance - min_total_distance)
        
        
        # Pour caque route => on ajoute pour les noeuds de cette route les metrics S5, S6, S7, S8 
        metric_S5 = self.average_width_per_route()                   # in cm 
        metric_S6 = self.average_span_per_route()                    # in radian 
        metric_S7 = self.average_compactness_per_route()
        metric_S8 = self.average_compactness_radian_per_route()
        
        
        for r in self.curr_routes:
        
            # mettre a jour les noeuds
            for i in range(len(r)):
                result[7][r[i]] = metric_S5
                result[8][r[i]] = metric_S6
                result[9][r[i]] = metric_S7
                result[10][r[i]] = metric_S8
            
        



        
        np.set_printoptions(threshold=np.inf)
        
        # print(f"""\n results for one environnement (one instance with multiple nodes) are as follows :\n
        #       x = {result[0]}, \n
        #       y = {result[1]}\n
        #       demand of customers = {result[2]}\n
        #       cumulative demande on the corresponding route = {result[3]}\n
        #       cumulative distance on the corresponding route = {result[4]}\n
        #       total demand on the corresponding route = {result[5]}\n
        #       total distance on the corresponding route = {result[6]}\n
        #       metric_S5 = {result[7]}\n
        #       metric_S6 = {result[8]}\n
        #       metric_S7 = {result[9]}\n
        #       metric_S8 = {result[10]}\n
        #       """)
        
        return np.transpose(np.array(result))
    
    
    
    def distance_point_to_line(self, point_G, node):
        # print(f" \n gravity_center in = distance_point_to_line {point_G} \n Node = {node} ({self.data[node, 0], self.data[node, 1]}) \n depot = ({self.data[0, 0], self.data[0, 1]})")
        #coordonés du point 
        x = self.data[node, 0]
        y = self.data[node, 1]
        
        #coordonés du depot
        D_x = self.data[0, 0]
        D_y = self.data[0, 1]
        
        #print(data["gravity_Centers"], idx_route)
        #cebtre de gravité
        G_x = point_G[0]
        G_y = point_G[1]
        
        if G_x == D_x and G_y == D_y : 
            # if the gravity center is the depot => we calculate the distance between the point and depot (center of gravity) beacause there is no line
            distance = math.sqrt((D_x - x)**2 + (D_y - y)**2)
            return distance
            
        else :
            # Calculate coefficients A, B, and C
            A = D_y - G_y
            B = G_x - D_x
            C = D_x * G_y - G_x * D_y
            
            # numerator = abs((x - D_x) * (G_y - D_y) - (y - D_y) * (G_x - D_x))
            # denominator = math.sqrt((G_x - D_x) ** 2 + (G_y - D_y) ** 2)   
            
            numerator = abs(A * x + B * y + C)
            denominator = math.sqrt(A**2 + B**2)
        
        # print(f" \n distance_point_to_line = {numerator / denominator} ; numerator = {numerator} / denominator = {denominator} \n with the gravity center = {point_G}")
        return numerator / denominator

    def distance_points_to_line(self, point_G, route):
        distance = []
        
        for i in route :
            # print(f"customer {i} = ({self.data[i, 0]}, {self.data[i, 0]})")
            distance.append(self.distance_point_to_line(point_G, i)) 
        return distance


    def gravity_center(self, route):
        # print(f"\n\n\n Calculate the gravity center of route {route}")
        # print(f"Depot ({self.data[0,0]}, {self.data[0,1]})")
        # for customer in route:
        #     print(f"customer {customer} ({self.data[customer,0]}, {self.data[customer,1]})")
            

        return  [(sum(self.data[customer,0] for customer in route) + self.data[0,0]) / (len(route)+1),
                 (sum(self.data[customer,1] for customer in route) + self.data[0,1]) / (len(route)+1)]
        
        # return  [(sum(self.data[customer,0] for customer in route)) / (len(route)),
        #          (sum(self.data[customer,1] for customer in route)) / (len(route))]
            
    
    def average_width_per_route(self):
        somme = 0
        for r in self.curr_routes:

            G_r = self.gravity_center(r) # returns a gravity center point (Gx, Gy)
            # print(f" \n gravity_center = {G_r}")
            # print(f"route = {r}")
            
            
            distance = self.distance_points_to_line(G_r, r)  # return vector of distances of each client with the line L crossing the gravity center and depot
            # print(f"\n distance = {distance}")
            max_d = max(distance)
            min_d = min(distance)
            somme = somme + max_d - min_d
            
        # print(f"\n somme = {somme}")
        # print(f"\n len(self.curr_routes) = {len(self.curr_routes)}")
        
        return somme / len(self.curr_routes)
            
    
    def radian_difference(self, point_1, point_2):
        # print(f"\n\n point_1 = {point_1};;   point_2 = {point_2}")
        #coordonés du depot
        D_x = self.data[0, 0]
        D_y = self.data[0, 0]
        
        #coordonés du point Ij 
        x1 = self.data[point_1, 0]
        y1 = self.data[point_1, 1]
        
        #coordonés du point Ik 
        x2 = self.data[point_2, 0]
        y2 = self.data[point_2, 1]
    
        # calculate the angle of each node with respect to the depot
        theta1 = math.atan2(y1 - D_y, x1 - D_x)
        theta2 = math.atan2(y2 - D_y, x2 - D_x)
        
        # Calculate the difference in angles
        delta_theta = theta2 - theta1 
        
        # Normalize the angle to be within the range [-pi, pi]
        if delta_theta > math.pi:
            delta_theta -= 2 * math.pi
        elif delta_theta < -math.pi:
            delta_theta += 2 * math.pi
            
        return delta_theta
    
        
    def radian(self, complete_r):
        # print(f"\n\n ERROR Radian ")
        radian = []
        
        for i in range(1, len(complete_r)):
            for j in range(1, len(complete_r)):
                delta_theta = self.radian_difference(complete_r[i], complete_r[j])

                radian.append(delta_theta)         
        
        return radian
    
    def average_span_per_route(self):    
        somme = 0
        for r in self.curr_routes:
            complete_r = [0]+r
            radian = self.radian(complete_r)  # return vector for each route of the diffrence between 2 nodes with respect to the depot D
            max_r = max(radian)
            somme = somme + max_r
        
        return somme / len(self.curr_routes)             
    
    def compactness(self, route):
        somme = 0
        #complete_r = [0] + route
        G_r = self.gravity_center(route)
        # print(f" \n gravity_center for compactness = {G_r}")
        for customer in route:
            # print(f"customer {customer} = ({self.data[customer, 0]}, {self.data[customer, 1]})")
            somme = somme + abs(self.distance_point_to_line(G_r, customer))
        
        return somme
        
        

        
    def average_compactness_per_route(self):
        
        # N  number of cutomers
        N  = len(self.data) - 1 
        somme = 0
        for r in self.curr_routes:
            # print(f"route in compactness = {r}")
            
            compactness = self.compactness(r)  # return vector for each route of the diffrence between 2 nodes with respect to the depot D
            somme = somme + compactness
        
            # print(f"\n compactness = {compactness}")
        
        # print(f"\n somme = {somme}")
        # print(f"\n N = {N}")
        return somme / N
        
    def radian_difference_with_cetre_de_gravité(self, G_r, point):
        #coordonés du depot
        D_x = self.data[0, 0]
        D_y = self.data[0, 0]
        
        #coordonés du point G_r (centre de gravité de la route)
        x1 = G_r[0]
        y1 = G_r[1]
        
        #coordonés du point Ik 
        x2 = self.data[point, 0]
        y2 = self.data[point, 1]
    
        # calculate the angle of each node with respect to the depot
        theta1 = math.atan2(y1 - D_y, x1 - D_x)
        theta2 = math.atan2(y2 - D_y, x2 - D_x)
        
        # Calculate the difference in angles
        delta_theta = theta2 - theta1 
        
        # Normalize the angle to be within the range [-pi, pi]
        if delta_theta > math.pi:
            delta_theta -= 2 * math.pi
        elif delta_theta < -math.pi:
            delta_theta += 2 * math.pi
            
        return delta_theta
    
    
    
    def radian_compactness(self, route):  
        # print(f"\n\n ERROR Radian_compactness ")
        somme = 0
        #complete_r = [0] + route
        G_r = self.gravity_center(route)
        
        for customer in route:
            somme = somme + self.radian_difference_with_cetre_de_gravité(G_r, customer)
        
        return somme 
    
    
    def average_compactness_radian_per_route(self):

        N  = len(self.data) - 1 
        somme = 0
        for r in self.curr_routes:
            
            compactness = self.radian_compactness(r)  # return vector for each route of the diffrence between 2 nodes with respect to the depot D
            somme = somme + compactness
        
        
        return somme / N
    
    
    
    
    def get_neighbours(self):
        n_vertices = self.distMat.shape[0]
        neighbours = []
        for i in range(n_vertices):
            index_dist = [(j, self.distMat[i][j]) for j in range(n_vertices)]
            sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
            neighbours.append([x[0] for x in sorted_index_dist])
        return neighbours
    
    def get_ruins(self, anchor, coefficient, n_ruins, ruined=[]):
        return find_ruins(n_ruins, anchor, coefficient, self.neighbours, self.curr_routes, ruined)
    
    def reward_func(self, old_d, new_d):
        # return max(-1, old_d-new_d)
        return old_d-new_d
        # return max(0, (old_d-new_d) /(old_d + 1e-8))
    
    def sim_annealing(self, new_dist, new_routes):
        b = False
        best_b = False
        reward = self.reward_func(self.curr_dist, new_dist)
        if new_dist<(self.curr_dist-self.T*np.log(np.random.random())):
            b = True
            if new_dist<self.best_dist:
                best_b = True
                self.best_dist = new_dist
                self.best_routes = copy.deepcopy(new_routes)
            self.curr_dist = new_dist
            self.curr_routes = new_routes
        self.T *= self.alpha_T
        return b, best_b, reward
    
    def __str__(self) -> str:
        return f" capacity = {self.cap} routes = {self.curr_routes} best_dist = {self.best_dist}, best_routes ={self.best_routes})"
    