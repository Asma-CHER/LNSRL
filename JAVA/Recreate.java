
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Recreate {

    public static void main(String[] args) throws IOException {
        // String data_root = "tmp_datas/";
        // String state_path = "states.txt";
        // String distMat_path = "distMat.txt";
        String data_root = args[0];
        String state_path = args[1];
        String distMat_path = args[2];

        StringBuilder result = new StringBuilder();

        String content = new String(Files.readAllBytes(Paths.get(state_path)), "UTF-8").replace("\r", ""); // MODIF :
                                                                                                           // Read
                                                                                                           // content
                                                                                                           // from
                                                                                                           // state_path
                                                                                                           // and remove
                                                                                                           // \r
                                                                                                           // characters;
        String[] lines = content.split("\n");
        ArrayList<Integer> batch_caps = new ArrayList<>();
        ArrayList<ArrayList<Integer>> batch_ruins = new ArrayList<>();
        ArrayList<ArrayList<int[]>> batch_routes = new ArrayList<>();
        for (String line : lines) {
            String[] line_split = line.split(":");

            ArrayList<Integer> ruins = new ArrayList<>();
            for (String r : line_split[1].split(","))
                ruins.add(Integer.parseInt(r));

            ArrayList<int[]> routes = new ArrayList<>();
            for (String route_str : line_split[2].split(";")) {
                String[] route_arr = route_str.split(",");
                int[] route = new int[route_arr.length];
                for (int i = 0; i < route_arr.length; i++) {
                    route[i] = Integer.parseInt(route_arr[i]);
                }
                routes.add(route);
            }

            batch_caps.add(Integer.parseInt(line_split[0]));
            batch_ruins.add(ruins);
            batch_routes.add(routes);
        }

        ArrayList<double[][]> datas = new ArrayList<>();
        for (int i = 0; i < lines.length; i++) {
            datas.add(Util_SimpleQuestionReader(data_root + "data_" + Integer.toString(i) + ".txt"));
        }

        ArrayList<double[][]> distance_matrices = new ArrayList<>();
        content = new String(Files.readAllBytes(Paths.get(distMat_path)), "UTF-8");
        lines = content.split("\n");
        for (String line : lines) {
            distance_matrices.add(Util_SimpleDistMatReader(line));
        }

        for (int i = 0; i < datas.size(); i++) {
            ArrayList<int[]> new_routes = step(datas.get(i),
                    batch_caps.get(i),
                    distance_matrices.get(i),
                    batch_ruins.get(i),
                    batch_routes.get(i));
            double new_distance = getRoutesDistance(new_routes, distance_matrices.get(i));
            result.append(new_distance + ":" + routes2str(new_routes) + "\n");
        }

        System.out.println(result.toString());
    }

    public static double[][] Util_SimpleDistMatReader(String DistMat) {
        String[] lines = DistMat.split(";");
        double[][] distance_matrix = new double[lines.length][lines.length];
        for (int i = 0; i < lines.length; i++) {
            String[] dists_str = lines[i].split(",");
            for (int j = 0; j < dists_str.length; j++) {
                distance_matrix[i][j] = Double.parseDouble(dists_str[j]);
            }
        }
        return distance_matrix;
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

    public static String routes2str(ArrayList<int[]> routes) {
        StringBuilder sb = new StringBuilder();
        for (int[] r : routes) {
            for (int i : r) {
                sb.append(i);
                sb.append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            sb.append(";");
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
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

    private static boolean checkValid(double[][] data, double[][] distance_matrix, int[] r, int c) {
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

    private static ArrayList<double[]> getValid(double[][] data, double[][] distance_matrix, int[] r, int c) {
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
            if (checkValid(data, distance_matrix, new_r, c)) {
                double new_dist = getRouteDistance(new_r, distance_matrix);
                valids.add(new double[] { i, new_dist - dist });
            }
            tmp_time += distance_matrix[curr_node][next_node];
            curr_node = i == r.length ? 0 : r[i];
        }
        return valids;
    }

    private static ArrayList<double[]> getValidCvrp(double[][] data, double[][] distance_matrix, int[] r, int c) {
        // Cette fonction prend en entrée les données (pour chaque noeud [x,y,
        // demand,0,0]) la matrice des distances, une route r et un noeud c
        // et elle retourne une liste de valids avec [i, gain] 'i' est la position et
        // 'gain' est la différence new_dist - dist

        ArrayList<double[]> valids = new ArrayList<>();
        double dist = getRouteDistance(r, distance_matrix); // calculer la distance de la route r

        for (int i = 0; i < (r.length + 1); i++) {

            // pour chaque position dans r on ajoute le noeud c et calcule la nouvelle
            // distance

            int[] new_r = insertNode(r, i, c);

            double new_dist = getRouteDistance(new_r, distance_matrix); // calculer la distance de la route new_r

            valids.add(new double[] { i, new_dist - dist }); // ajouter dans valids un vecteur de 2 valeur 1) i : la
                                                             // position; 2) le gain en distance

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

    public static ArrayList<int[]> regretRepair(double[][] data, double capcity, double[][] distance_matrix,
            ArrayList<int[]> absent_route, ArrayList<Integer> absents) {

        StringBuilder result_ = new StringBuilder();

        result_.append("REPAIR Regret ");

        while (!absents.isEmpty()) {

            List<Object[]> regrets = costRegrets(data, capcity, distance_matrix, absents, absent_route);
            
            for (Object[] array : regrets) {
                result_.append(" customer = ").append(array[0]).append("  ;  ");
                result_.append(" regret = ").append(array[1]).append("  ;  ");
                result_.append(" route = ").append(Arrays.toString((int[]) array[2])).append("  ;  ");
                result_.append(" idx = ").append(array[3]).append("  ;  ");
                
                
            }

            int customer;

            if (!regrets.isEmpty()) {

                Object[] regretChoice = regrets.remove(regrets.size() - 1); // pop the last item [customer, regret,
                result_.append("\n regretChoice "); // route, idx]

                result_.append(" customer = ").append(regretChoice[0]).append("  ::  ");
                result_.append(" regret = ").append(regretChoice[1]).append("  ::  ");
                result_.append(" route = ").append(Arrays.toString((int[]) regretChoice[2])).append("  ::  ");
                result_.append(" idx = ").append(regretChoice[3]).append("  ::  ");

                result_.append("\n");

                customer = (int) regretChoice[0];
                int[] route = (int[]) regretChoice[2];
                int idx = (int) regretChoice[3];
                int routeIdx = -1;
                routeIdx = absent_route.indexOf(route);

                result_.append("\n customer " + customer + "; route " + Arrays.toString((int[]) route) + "; idx " + idx
                        + "; routeIdx "
                        + routeIdx + " \n ");

                if (routeIdx != -1) {

                    result_.append("\n routeIdx = " + routeIdx + " \n ");
                    int[] newRoute = new int[route.length + 1];
                    System.arraycopy(route, 0, newRoute, 0, idx);
                    newRoute[idx] = customer;
                    System.arraycopy(route, idx, newRoute, idx + 1, route.length - idx);
                    absent_route.set(routeIdx, newRoute);
                } else {
                    result_.append("\n There is no routeIdx for customer = " + customer + "\n ");
                }

            } else {
                customer = (int) absents.get(0);
                absent_route.add(new int[] { customer });
            }

            // System.out.println("\n\n Remove " + customer + " from " +
            // absents.toString());
            absents.remove(Integer.valueOf(customer));

        }

        // System.out.println(result_.toString());
        if (TotalDemand(data, absent_route, capcity)) {
            System.out.println("absent_route = " + absent_route);
        }

        return absent_route;
    }

    private static boolean TotalDemand(double[][] data, ArrayList<int[]> routes, double capacity) {
        double total_d = 0;

        for (int[] route : routes) {
            total_d = 0;
            for (int customer : route) {
                total_d += data[customer][2];
            }
            if (total_d > capacity) {
                return true;
            }
        }
        return false;

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

    // THIS IS REGRET REPAIR
    private static ArrayList<int[]> recreate_regret_repair(double[][] data, double capcity, double[][] distance_matrix,
            ArrayList<int[]> absent_route, ArrayList<Integer> absents, int lastLength) {

        // System.out.println("\n\n ********* HERE PRINT entree Recreate \n");
        // System.out.println("data = " + Arrays.deepToString(data));
        // System.out.println("*********capcity = " + capcity);

        // for (int i = 0; i < absent_route.size(); i++) {
        // System.out.print(Arrays.toString(absent_route.get(i)));
        // if (i < absent_route.size() - 1) {
        // System.out.print(", ");
        // }
        // }
        // System.out.println("]");

        // System.out.println("absents = " + absents.toString());
        // System.out.println("lastLength = " + lastLength);
        // System.out.println(" \n\n ");

        ArrayList<int[]> current_route = regretRepair(data, capcity, distance_matrix,
                absent_route, absents);

        // System.out.println("\n \n ********* AFTER REPAIR current_route = " + " \n ");
        // for (int i = 0; i < current_route.size(); i++) {
        // System.out.print(Arrays.toString(current_route.get(i)));
        // if (i < current_route.size() - 1) {
        // System.out.print(", ");
        // }
        // }
        return current_route;
    }

    // THIS IS BEST INSERT
    private static ArrayList<int[]> recreate(double[][] data, double capcity, double[][] distance_matrix,
            ArrayList<int[]> absent_route, ArrayList<Integer> absents, int lastLength) {

        // StringBuilder sb = new StringBuilder();
        // sb.append("HERE PRINT entree Recreate: ").append("\n");
        // sb.append(" data = ").append(convertDoubleArrayToString(data)).append("\n")
        // .append(" capcity:").append(capcity).append("\n")
        // .append(" distance_matrix =
        // ").append(convertDoubleArrayToString(distance_matrix)).append("\n");

        // // Impression de la chaîne finale
        // System.out.println(sb.toString());

        // System.out.println("\n\n HERE PRINT entree Recreate \n");
        // System.out.println("data = " + Arrays.deepToString(data));
        // System.out.println("*********capcity = " + capcity);
        // System.out.println("distance_matrix = " +
        // Arrays.deepToString(distance_matrix));

        // System.out.println("absent_route = " + Arrays.deepToString(absent_route));
        // System.out.print("absent_route = [");
        // for (int i = 0; i < absent_route.size(); i++) {
        // System.out.print(Arrays.toString(absent_route.get(i)));
        // if (i < absent_route.size() - 1) {
        // System.out.print(", ");
        // }
        // }
        // System.out.println("]");

        // System.out.println("absents = " + absents.toString());
        // System.out.println("lastLength = " + lastLength);
        // System.out.println(" \n\n ");

        Collections.shuffle(absents);
        ArrayList<Integer> newAbsents = new ArrayList<>();
        ArrayList<Integer> toKeep = new ArrayList<>();
        ArrayList<int[]> current_route = absent_route;

        for (int i = 0; i < absents.size(); i++) {
            // pour chaque node c absents (càd qui n'ai pas affecté à une route)
            // on essai de trouver the best place

            int c = absents.get(i);
            ArrayList<double[]> probablePlace = new ArrayList<>();

            for (int ir = 0; ir < absent_route.size(); ir++) {
                // we go through the current routes one by one

                int[] r = absent_route.get(ir);

                // vérifier si on ajoute le node on dépasse pas la capacité
                double dmd_sum = 0;
                for (int _tmp_node : r)
                    dmd_sum += data[_tmp_node][2];

                if ((dmd_sum + data[c][2]) > capcity)
                    continue;

                // all possible int values can round-trip to a double safely.
                // recupérer une liste de vecteurs; chaque vecteur contient 2 valeurs :
                // (i : l'indice ou la position dans cette route, et le gain en distance en
                // ajoutant le noeud dans cette position)
                ArrayList<double[]> valids = getValidCvrp(data, distance_matrix, r, c);
                for (double[] v : valids)
                    probablePlace.add(new double[] { ir, v[0], v[1] }); // pour chaque vecteur dans valids on créé
                                                                        // une
                                                                        // place probable pour cette route ir
            }

            // donc ici pour le noeud c on a toutes les positions dans differente route avec
            // le gain (new_dist - dist)

            // trier les position selon le gain content[2]
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
                adding_pos = probablePlace.get(0); // récupérer la meilleure position
            } else if (lastLength > 0 && lastLength <= current_route.size()) {
                toKeep.add(i);
                continue;
            }
            current_route = route_add(current_route, c, adding_pos); // si adding_pos == { -1, -1, -1 } alors pas de
                                                                     // position valide => pénalisation
        }

        // System.out.println("\n toKeep size = " + toKeep.size() + " list = " + toKeep
        // + " \n ");

        for (int i : toKeep) {
            // System.out.println("\n IN toKeep \n ");
            newAbsents.add(absents.get(i));
        }
        absents.clear();

        // System.out.println("\n newAbsents size = " + newAbsents.size() + " list = " +
        // newAbsents + " \n ");

        for (int i : newAbsents) {
            // System.out.println("\n IN newAbsents \n ");
            absents.add(i);
        }

        return current_route;
    }

    public static ArrayList<int[]> step(double[][] data, double cap, double[][] distance_matrix,
            ArrayList<Integer> ruins, ArrayList<int[]> init_route) {
        ArrayList<int[]> absent_route = ruin_routeSummary(init_route, ruins);
        return recreate_regret_repair(data, cap, distance_matrix, absent_route, ruins, 0);
        // return recreate(data, cap, distance_matrix, absent_route, ruins, 0);
    }

}
