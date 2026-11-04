
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.nio.file.Files;
import java.nio.file.Paths;

public class RandomSolver {

    private static int m = 1;

    public static void main(String[] args) throws IOException {
        double[][] data = Util_SimpleQuestionReader(args[0]);
        int capacity = Integer.parseInt(args[1]);
        int n_iter = Integer.parseInt(args[2]);
        int n_ruin = Integer.parseInt(args[3]);

        ArrayList<int[]> routes = solveRAND_VRP(data, capacity, n_iter, n_ruin, null, 100.0f, 1.0f, 0);
        double distance = getRoutesDistance(routes, getDistanceMatrix(data));
        StringBuilder sb = new StringBuilder();
        sb.append(distance);
        sb.append(":");
        for (int[] r : routes) {
            for (int i : r) {
                sb.append(i);
                sb.append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            sb.append(";");
        }
        sb.deleteCharAt(sb.length() - 1);
        System.out.println(sb.toString());
    }

    public static double[][] Util_SimpleQuestionReader(String path) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(path)), "UTF-8");
        String[] lines = content.split("\n");
        double[][] data = new double[lines.length][6];
        for (int i = 0; i < lines.length; i++) {
            String[] d = lines[i].split("\\s+");
            for (int j = 0; j < d.length; j++) {
                data[i][j] = Double.parseDouble(d[j]);
            }
        }
        return data;
    }

    private static int indexOfMin(double[] a) {
        int loc = 0;
        double min = a[loc];
        for (int i = 1; i < a.length; i++) {
            if (a[i] < min) {
                min = a[i];
                loc = i;
            }
        }
        return loc;
    }

    public static void sortRoutes(ArrayList<int[]> routes) {
        Collections.sort(routes, new Comparator<int[]>() {
            public int compare(int[] content0, int[] content1) {
                if (content0[0] < content1[0]) {
                    return -1;
                } else if (content0[0] > content1[0]) {
                    return 1;
                }
                return 0;
            }
        });
    }

    public static boolean compareRoutes(ArrayList<int[]> routesA, ArrayList<int[]> routesB) {
        if (routesA.size() != routesB.size())
            return false;

        ArrayList<int[]> routesAReformat = new ArrayList<>();
        ArrayList<int[]> routesBReformat = new ArrayList<>();

        for (int i = 0; i < routesA.size(); i++) {
            HashSet<Integer> set1 = new HashSet<Integer>(
                    Arrays.stream(routesA.get(i)).boxed().collect(Collectors.toList()));
            HashSet<Integer> set2 = new HashSet<Integer>(
                    Arrays.stream(routesB.get(i)).boxed().collect(Collectors.toList()));
            routesAReformat.add(set1.stream().mapToInt(Number::intValue).toArray());
            routesBReformat.add(set2.stream().mapToInt(Number::intValue).toArray());
        }
        sortRoutes(routesAReformat);
        sortRoutes(routesBReformat);
        for (int i = 0; i < routesAReformat.size(); i++) {
            if (routesAReformat.get(i).length != routesBReformat.get(i).length) {
                System.out.println(" -- Route A: " + java.util.Arrays.toString(routesAReformat.get(i)));
                System.out.println(" -- Route B: " + java.util.Arrays.toString(routesBReformat.get(i)));
                return false;
            }
            for (int j = 0; j < routesAReformat.get(i).length; j++) {
                if (routesAReformat.get(i)[j] != routesBReformat.get(i)[j]) {
                    System.out.println(" -- Route A: " + java.util.Arrays.toString(routesAReformat.get(i)));
                    System.out.println(" -- Route B: " + java.util.Arrays.toString(routesBReformat.get(i)));
                    return false;
                }
            }
        }
        return true;
    }

    public static double getDistance(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
    }

    // 0 is not included in the route but will be calculated.
    public static double getRouteDistance(int[] route, double[][] distance_matrix) {
        int last_node = 0;
        double distance = 0.0;
        for (int c : route) {
            distance += distance_matrix[last_node][c];
            last_node = c;
        }
        distance += distance_matrix[last_node][0];
        return distance;
    }

    // 0 is not included in routes but will be calculated.
    public static double getRoutesDistance(ArrayList<int[]> routes, double[][] distance_matrix) {
        double total_distance = 0.0;
        for (int[] r : routes) {
            total_distance += getRouteDistance(r, distance_matrix);
        }
        return total_distance;
    }

    public static double[][] getDistanceMatrix(double[][] data) {
        double[][] distance_matrix = new double[data.length][data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                if (i != j)
                    distance_matrix[i][j] = getDistance(data[i][0], data[i][1], data[j][0], data[j][1]);
            }
        }
        return distance_matrix;
    }

    private static int[][] buildNeighbours(double[][] distance_matrix) {
        int[][] neighbours = new int[distance_matrix.length][distance_matrix.length];
        for (int i = 0; i < distance_matrix.length; i++) {
            ArrayIndexComparator comparator = new ArrayIndexComparator(distance_matrix[i], false);
            Integer[] indexes = comparator.createIndexArray();
            Arrays.sort(indexes, comparator);
            neighbours[i] = Arrays.stream(indexes).mapToInt(Integer::intValue).toArray();
        }
        return neighbours;
    }

    private static int find_r(ArrayList<int[]> lastRoute, int c) {
        int index = -1;
        for (int i = 0; i < lastRoute.size(); i++) {
            if (Arrays.stream(lastRoute.get(i)).anyMatch(j -> j == c)) {
                index = i;
                break;
            }
        }
        return index;
    }

    private static int[] string_remove(int[] r, int l_t, int c) {
        int i_c = Arrays.stream(r).boxed().collect(Collectors.toList()).indexOf(c);
        int range1 = Math.max(0, i_c + 1 - l_t);
        int range2 = Math.min(i_c, r.length - l_t) + 1;
        int start = ThreadLocalRandom.current().nextInt(range1, range2);
        return Arrays.copyOfRange(r, start, start + l_t);
    }

    private static int[] split_remove(int[] r, int l_t, int c) {
        int additional_l = Math.min(m, r.length - l_t);
        int l_t_m = l_t + additional_l;
        int i_c = Arrays.stream(r).boxed().collect(Collectors.toList()).indexOf(c);
        int range1 = Math.max(0, i_c + 1 - l_t_m);
        int range2 = Math.min(i_c, r.length - l_t_m) + 1;
        int start = ThreadLocalRandom.current().nextInt(range1, range2);
        List<Integer> potential_removal = new ArrayList<Integer>(l_t_m);
        for (int i : Arrays.copyOfRange(r, start, start + l_t_m))
            potential_removal.add(i);
        Collections.shuffle(potential_removal);
        return potential_removal.subList(0, l_t).stream().mapToInt(i -> i).toArray();
    }

    private static int[] remove_nodes(int[] r, int l_t, int c, double m_alpha) {
        int[] removed;
        if (Math.random() < 0.5) {
            removed = string_remove(r, l_t, c);
        } else {
            removed = split_remove(r, l_t, c);
            if (m < (r.length - l_t) || Math.random() > m_alpha)
                m++;
        }
        return removed;
    }

    public static ArrayList<Integer> sample(int n_total, int n_sample) {
        // if (n_sample>n_total) throw new ValueException("n_sample larger than
        // n_total.");
        ArrayList<Integer> result = new ArrayList<>();
        ArrayList<Integer> probables = new ArrayList<>();
        for (int i = 0; i < n_total; i++)
            probables.add(i);
        Random random = new Random();
        for (int i = 0; i < n_sample; i++) {
            int index = random.nextInt(probables.size());
            result.add(probables.get(index));
            probables.remove(index);
        }
        return result;
    }

    private static ArrayList<Integer> ruin_getAbsents(double[][] data, int n_ruin) {
        ArrayList<Integer> ruined = new ArrayList<Integer>();
        for (int i : sample(data.length - 1, n_ruin)) { // sample : selects randomly n_ruin node in liste from '0' to
                                                        // 'data.length-1'
            ruined.add(i + 1);
        }
        return ruined;
    }

    private static ArrayList<int[]> ruin_routeSummary(ArrayList<int[]> lastRoute, ArrayList<Integer> absents) {
        ArrayList<int[]> absentRoute = new ArrayList<int[]>();
        for (int i = 0; i < lastRoute.size(); i++) {
            ArrayList<Integer> r = new ArrayList<>();
            for (int c : lastRoute.get(i)) {
                if (!absents.contains(c))
                    r.add(c);
            }
            if (r.size() > 0) {
                absentRoute.add(r.stream().mapToInt(j -> j).toArray());
            }
        }
        return absentRoute;
    }

    private static int[] insertNode(int[] old_r, int pos, int c) {
        int[] new_r = new int[old_r.length + 1];
        for (int i = 0; i < new_r.length; i++) {
            if (i < pos) {
                new_r[i] = old_r[i];
            } else if (i > pos) {
                new_r[i] = old_r[i - 1];
            } else {
                new_r[i] = c;
            }
        }
        return new_r;
    }

    // VRPTW

    private static boolean checkValid_(double[][] data, double[][] distance_matrix, int[] r, int c) {
        double time_current = 0;
        int curr_node = 0;
        for (int i = 0; i < (r.length + 1); i++) {
            int next_node = i == r.length ? 0 : r[i];
            time_current += distance_matrix[curr_node][next_node];
            time_current = Math.max(data[next_node][3], time_current);
            if (time_current <= data[next_node][4]) {
                time_current += data[next_node][5];
            } else {
                return false;
            }
            curr_node = i == r.length ? 0 : r[i];
        }
        return true;
    }

    // VRPTW
    private static ArrayList<double[]> getValid_(double[][] data, double[][] distance_matrix, int[] r, int c) {
        ArrayList<double[]> valids = new ArrayList<>();
        double dist = getRouteDistance(r, distance_matrix);
        double tmp_time = 0;
        int curr_node = 0;
        for (int i = 0; i < (r.length + 1); i++) {
            int next_node = i == r.length ? 0 : r[i];
            tmp_time = Math.max(tmp_time, data[curr_node][3]);
            tmp_time += data[curr_node][5];
            if (tmp_time + distance_matrix[curr_node][c] > data[c][4])
                break;
            int[] new_r = insertNode(r, i, c);
            if (checkValid_(data, distance_matrix, new_r, c)) {
                double new_dist = getRouteDistance(new_r, distance_matrix);
                valids.add(new double[] { i, new_dist - dist });
            }
            tmp_time += distance_matrix[curr_node][next_node];
            curr_node = i == r.length ? 0 : r[i];
        }
        return valids;
    }

    private static ArrayList<double[]> getValidCvrp(double[][] data, double[][] distance_matrix, int[] r, int c) {
        ArrayList<double[]> valids = new ArrayList<>();
        double dist = getRouteDistance(r, distance_matrix);

        for (int i = 0; i < (r.length + 1); i++) {

            int[] new_r = insertNode(r, i, c);

            double new_dist = getRouteDistance(new_r, distance_matrix);

            valids.add(new double[] { i, new_dist - dist });

        }
        return valids;
    }

    private static ArrayList<int[]> route_add(ArrayList<int[]> absent_route, int c, double[] adding_pos) {
        if (adding_pos[0] == -1) {
            absent_route.add(new int[] { c });
            return absent_route;
        }
        int[] new_r = insertNode(absent_route.get((int) adding_pos[0]), (int) adding_pos[1], c);
        absent_route.set((int) adding_pos[0], new_r);
        return absent_route;
    }

    private static ArrayList<int[]> recreate(double[][] data, double capcity, double[][] distance_matrix,
            ArrayList<int[]> absent_route, ArrayList<Integer> absents, int lastLength) {
        // 算子II
        Collections.shuffle(absents);

        ArrayList<Integer> newAbsents = new ArrayList<>();
        ArrayList<Integer> toKeep = new ArrayList<>();
        ArrayList<int[]> current_route = absent_route;
        // System.out.println("\n absents = " + absents);
        for (int i = 0; i < absents.size(); i++) {
            int c = absents.get(i);
            ArrayList<double[]> probablePlace = new ArrayList<>();
            for (int ir = 0; ir < absent_route.size(); ir++) {
                int[] r = absent_route.get(ir);
                double dmd_sum = 0;
                for (int _tmp_node : r)
                    dmd_sum += data[_tmp_node][2];

                if ((dmd_sum + data[c][2]) > capcity)
                    // {
                    // System.out.println(
                    // "\n Here NOT POSSIBLE check cap and demand ; cap = " + capcity + " dmd_sum "
                    // + dmd_sum
                    // + " le noeud demand " + data[c][2]);
                    continue;
                // } else {
                // System.out.println("\n Here check cap and demand ; cap = " + capcity + "
                // dmd_sum " + dmd_sum
                // + " le noeud demand " + data[c][2]);
                // }

                // all possible int values can round-trip to a double safely.
                ArrayList<double[]> valids = getValidCvrp(data, distance_matrix, r, c);
                for (double[] v : valids)
                    probablePlace.add(new double[] { ir, v[0], v[1] });
            }

            double[] adding_pos = new double[] { -1, -1, -1 };
            if (probablePlace.size() > 0) {
                Collections.sort(probablePlace, new Comparator<double[]>() {
                    public int compare(double[] content0, double[] content1) {
                        if (content0[2] < content1[2]) {
                            return -1;
                        } else if (content0[2] > content1[2]) {
                            return 1;
                        }
                        return 0;
                    }
                });
                adding_pos = probablePlace.get(0);
            } else if (lastLength > 0 && lastLength <= current_route.size()) {
                toKeep.add(i);
                continue;
            }

            current_route = route_add(current_route, c, adding_pos);
        }
        for (int i : toKeep)
            newAbsents.add(absents.get(i));
        absents.clear();
        for (int i : newAbsents)
            absents.add(i);
        return current_route;
    }

    public static ArrayList<int[]> regretRepair(double[][] data, double capcity, double[][] distance_matrix,
            ArrayList<int[]> absent_route, ArrayList<Integer> absents) {

        while (!absents.isEmpty()) {

            List<Object[]> regrets = costRegrets(data, capcity, distance_matrix, absents, absent_route);

            int customer;

            if (!regrets.isEmpty()) {

                Object[] regretChoice = regrets.remove(regrets.size() - 1); // pop the last item [customer, regret,

                customer = (int) regretChoice[0];
                int[] route = (int[]) regretChoice[2];
                int idx = (int) regretChoice[3];
                int routeIdx = -1;
                routeIdx = absent_route.indexOf(route);

                // insérer customer dans la route à la position idx
                if (routeIdx != -1) {

                    int[] newRoute = new int[route.length + 1];
                    System.arraycopy(route, 0, newRoute, 0, idx);
                    newRoute[idx] = customer;
                    System.arraycopy(route, idx, newRoute, idx + 1, route.length - idx);
                    absent_route.set(routeIdx, newRoute);
                }

            } else {
                customer = (int) absents.get(0);
                absent_route.add(new int[] { customer });
            }

            absents.remove(Integer.valueOf(customer));

        }

        return absent_route;
    }

    private static List<Object[]> costRegrets(double[][] data, double capcity, double[][] distance_matrix,
            ArrayList<Integer> absents, ArrayList<int[]> absent_route) {
        List<Object[]> regrets = new ArrayList<>();

        for (int customer : absents) {
            Object[] regret = twoRegret(data, capcity, distance_matrix, customer, absent_route); // regret=[regret,
                                                                                                 // route, idx]

            if (regret[1] != null) {
                regrets.add(new Object[] { customer, regret[0], regret[1], regret[2] });
            }
        }

        if (!regrets.isEmpty()) {
            regrets.sort(Comparator.comparingDouble(r -> ((double) r[1])));
        }

        return regrets;
    }

    private static Object[] twoRegret(double[][] data, double capcity, double[][] distance_matrix, int customer,
            ArrayList<int[]> absent_route) {
        List<Object[]> costs = new ArrayList<>();
        double regret = 0;

        for (int routeIdx = 0; routeIdx < absent_route.size(); routeIdx++) {
            int[] route = absent_route.get(routeIdx);
            if (canInsert(data, capcity, customer, route)) {
                for (int idx = 0; idx <= route.length; idx++) {
                    Double cost = insertCost(distance_matrix, customer, route, idx);
                    costs.add(new Object[] { cost, route, idx });
                }
            }
        }

        if (costs.size() > 0) {
            costs.sort(Comparator.comparingDouble(c -> (double) c[0])); // trie ascendant
            if (costs.size() > 1) {
                regret = (double) costs.get(1)[0] - (double) costs.get(0)[0]; // donc ici normalement le regret est
                                                                              // positif

            } else {
                regret = (double) costs.get(0)[0];
            }
            return new Object[] { regret, costs.get(0)[1], costs.get(0)[2] };
        }

        return new Object[] { null, null, null };
    }

    private static boolean canInsert(double[][] data, double capacity, int customer, int[] route) {
        int total = 0;
        for (int node : route) {

            total += data[node][2];
        }
        total += data[customer][2];
        return total <= capacity;
    }

    private static Double insertCost(double[][] distance_matrix, int customer, int[] route, int idx) {
        int pred = (idx == 0) ? 0 : route[idx - 1];
        int succ = (idx == route.length) ? 0 : route[idx];

        // Increase in cost of adding customer, minus cost of removing old edge
        return distance_matrix[pred][customer] + distance_matrix[customer][succ] - distance_matrix[pred][succ];
    }

    // Regret repair operator
    private static ArrayList<int[]> recreate_regret_repair(double[][] data, double capacity, double[][] distance_matrix,
            ArrayList<int[]> absente_route, ArrayList<Integer> absents, int lastLength) {

        ArrayList<int[]> current_route = regretRepair(data, capacity, distance_matrix, absente_route, absents);
        return current_route;
    }

    public static ArrayList<int[]> solveRAND_VRP(double[][] data, int capacity, int n_iter) {
        return solveRAND_VRP(data, capacity, n_iter, data.length / 10, null, 100.0f, 1.0f, 0);
    }

    // Data Format: XCOORD., YCOORD., DEMAND, READY TIME, DUE TIME, SERVICE TIME =>
    // changé to XCOORD., YCOORD., DEMAND
    public static ArrayList<int[]> solveRAND_VRP(double[][] data, double cap,
            int n_iter, int n_ruin,
            ArrayList<int[]> init_route,
            double init_T, double final_T,
            int verbose_step) {
        // Calculate the distance matrix.
        double[][] distance_matrix = getDistanceMatrix(data);

        // The 0 point (depot) should not be included in the routes. => i starts in 1
        ArrayList<int[]> best_route;
        if (init_route == null) {
            best_route = new ArrayList<int[]>();
            for (int i = 1; i < data.length; i++) {
                best_route.add(new int[] { i });
            }
            ArrayList<Integer> absents = new ArrayList<Integer>();
            for (int i = 1; i < data.length; i++) {
                absents.add(i);
            }
            ArrayList<int[]> absent_route = ruin_routeSummary(best_route, absents);
            best_route = recreate_regret_repair(data, cap, distance_matrix, absent_route, absents, 0);
        } else {
            best_route = init_route;
        }
        if (n_iter <= 0)
            return best_route;
        double best_distance = getRoutesDistance(best_route, distance_matrix);
        ArrayList<int[]> last_route = best_route;
        double last_distance = best_distance;

        // Get neighbours matrix : pour chaque ligne i on a les voisins de i triés (du
        // plus proche au plus loin)
        int[][] neighbours = buildNeighbours(distance_matrix); // ne sert à rien dans le random operator
        double alpha_T = Math.pow((final_T / init_T), (1.0 / n_iter));
        double temperature = init_T;

        for (int i_iter = 0; i_iter < n_iter; i_iter++) {

            // Ruin operator (random)
            // ruin_getAbsents retourn la liste des nodes a supprimer (choisi aléatoirement)
            ArrayList<Integer> absents = ruin_getAbsents(data, n_ruin);

            // Remove nodes.
            // ruin_routeSummary returns the destroyed routes
            ArrayList<int[]> absent_route = ruin_routeSummary(last_route, absents);

            // Recreate
            // repair 1) recreate : best repair; 2) recreate_regret_repair : regret repair
            ArrayList<int[]> current_route = recreate_regret_repair(data, cap, distance_matrix,
                    absent_route, absents, 0);
            // ArrayList<int[]> current_route = recreate_regret_repair(data, cap,
            // distance_matrix, absent_route, absents, 0);

            // Check and update
            double current_distance = getRoutesDistance(current_route, distance_matrix);
            // if (current_route.size()<last_route.size() ||
            // (current_route.size()==last_route.size() &&
            // current_distance<(last_distance-temperature*Math.log(Math.random())))) {
            if (current_distance < (last_distance - temperature * Math.log(Math.random()))) {
                last_route = current_route;
                last_distance = current_distance;
                if (current_route.size() < best_route.size() || current_distance < best_distance) {
                    best_route = current_route;
                    best_distance = current_distance;
                }
            }
            temperature *= alpha_T;

            if (verbose_step > 0 && ((i_iter + 1) % verbose_step == 0 || (i_iter + 1) == n_iter)) {
                System.out.println("# iter: " + (i_iter + 1) + "/" + n_iter);
                System.out.println("    --best route: " + best_route.size() + ", " + best_distance);
                System.out.println("    --last route: " + last_route.size() + ", " + last_distance);
                System.out.println("    --curr route: " + current_route.size() + ", " + current_distance);
            }

        }

        return best_route;
    }

}
