import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

/**
 * Persistent interface to the unchanged Recreate.step repair operator.
 *
 * The instance and distance matrix are loaded once. Each subsequent input line
 * contains capacity:ruined-customers:routes and produces one distance:routes
 * response. This removes per-iteration JVM startup and filesystem I/O.
 */
public final class RepairWorker {

    private RepairWorker() {
    }

    private static ArrayList<Integer> parseRuins(String value) {
        ArrayList<Integer> ruins = new ArrayList<>();
        if (value.isEmpty()) {
            return ruins;
        }
        for (String node : value.split(",")) {
            ruins.add(Integer.parseInt(node));
        }
        return ruins;
    }

    private static ArrayList<int[]> parseRoutes(String value) {
        ArrayList<int[]> routes = new ArrayList<>();
        if (value.isEmpty()) {
            return routes;
        }
        for (String routeValue : value.split(";")) {
            String[] nodes = routeValue.split(",");
            int[] route = new int[nodes.length];
            for (int index = 0; index < nodes.length; index++) {
                route[index] = Integer.parseInt(nodes[index]);
            }
            routes.add(route);
        }
        return routes;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException(
                    "Usage: RepairWorker <instance-data.txt> <distance-matrix.txt>");
        }

        double[][] data = Recreate.Util_SimpleQuestionReader(args[0]);
        String distanceLine = Files.readAllLines(
                Paths.get(args[1]), StandardCharsets.UTF_8).get(0);
        double[][] distanceMatrix = Recreate.Util_SimpleDistMatReader(distanceLine);

        try (BufferedReader input = new BufferedReader(
                     new InputStreamReader(System.in, StandardCharsets.UTF_8));
             BufferedWriter output = new BufferedWriter(
                     new OutputStreamWriter(System.out, StandardCharsets.UTF_8))) {
            String line;
            while ((line = input.readLine()) != null) {
                if (line.equals("QUIT")) {
                    break;
                }
                try {
                    String[] parts = line.split(":", -1);
                    if (parts.length != 3) {
                        throw new IllegalArgumentException("Invalid repair request");
                    }
                    double capacity = Double.parseDouble(parts[0]);
                    ArrayList<Integer> ruins = parseRuins(parts[1]);
                    ArrayList<int[]> routes = parseRoutes(parts[2]);
                    ArrayList<int[]> repaired = Recreate.step(
                            data, capacity, distanceMatrix, ruins, routes);
                    double distance = Recreate.getRoutesDistance(repaired, distanceMatrix);
                    output.write(distance + ":" + Recreate.routes2str(repaired));
                    output.newLine();
                    output.flush();
                } catch (Exception error) {
                    error.printStackTrace(System.err);
                    output.write("ERROR:" + error.getClass().getSimpleName()
                            + ":" + String.valueOf(error.getMessage()));
                    output.newLine();
                    output.flush();
                }
            }
        }
    }
}
