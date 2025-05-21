# section TRIMET BUS THINGS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

from phart import ASCIIRenderer
from bs4 import BeautifulSoup

from datetime import datetime
import networkx as nx
import requests
import pandas as pd
import matplotlib.pyplot as plt

import math


def get_direction(x1, y1, x2, y2):
    """
    Returns the cardinal direction (e.g., 'North', 'Southwest')
    from (x1, y1) to (x2, y2).
    """
    # Calculate angle in degrees
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle = (angle + 360) % 360  # Normalize to [0, 360)

    # Define full-text directions
    directions = [
        (0, "east"),
        (45, "northeast"),
        (90, "north"),
        (135, "northwest"),
        (180, "west"),
        (225, "southwest"),
        (270, "south"),
        (315, "southeast"),
        (360, "east"),
    ]

    # Find the closest direction
    for i in range(len(directions) - 1):
        lower_angle, dir_name = directions[i]
        upper_angle, _ = directions[i + 1]
        if lower_angle <= angle < upper_angle:
            return dir_name

    return "East"  # Fallback


def get_df(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table")

    df = pd.read_html(str(table))
    return df[0]


def df_to_travel_times(df):
    """THIS NEEDS TO HAVE THE FIRST ROW POPULATED"""
    df.columns = [d.split(" Stop ID")[0] for d in df.columns]
    col_times = dict(
        zip(df.columns, [time.strptime("0" + l[:4], "%H:%M") for l in df.iloc[0]])
    )
    return {k: datetime(*v[:5]) for k, v in col_times.items()}


# time_dict = df_to_travel_times(get_df("https://trimet.org/schedules/w/t1017_0.htm"))
def time_dict_to_digraph(stops):
    G = nx.DiGraph()

    # Create edges with time differences (in minutes)
    stop_names = list(stops.keys())

    for i in range(len(stop_names) - 1):
        start = stop_names[i]
        end = stop_names[i + 1]
        delta = (stops[end] - stops[start]).total_seconds() / 60  # in minutes
        G.add_edge(start, end, weight=delta)
    return G


def get_stops_from_map(bus_number) -> list:
    # Optional: Set up Chrome options (e.g., headless mode)
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run in headless mode

    # Path to your ChromeDriver (make sure it's in PATH or provide full path)
    driver = webdriver.Chrome(options=options)

    time.sleep(6)
    url = f"https://ride.trimet.org/home/route/{bus_number}/"
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    elements = soup.find_all(
        lambda tag: tag.has_attr("class")
        and any(cls.startswith("styled__StopListBody") for cls in tag["class"])
    )

    element = elements[0]
    buttons = element.find_all("button")

    stops = [button.text.strip() for button in buttons]
    return stops


# CHAT GPT CODE


def insert_all_stops(G, stop_sequence):
    G_augmented = nx.DiGraph()

    i = 0
    while i < len(stop_sequence) - 1:
        current = stop_sequence[i]

        # Search for next known node in original graph
        j = i + 1
        while j < len(stop_sequence) and stop_sequence[j] not in G.nodes:
            j += 1

        if j >= len(stop_sequence):
            break

        next_known = stop_sequence[j]

        # Check if the original graph has an edge between current and next_known
        if G.has_edge(current, next_known):
            original_weight = G[current][next_known]["weight"]
        else:
            # Try to look for a path through G
            try:
                original_weight = nx.shortest_path_length(
                    G, current, next_known, weight="weight"
                )
            except:
                # If there's no path, skip
                i = j
                continue

        # Get the full segment: current -> intermediate stops -> next_known
        segment = stop_sequence[i : j + 1]
        num_edges = len(segment) - 1
        weight_per_edge = original_weight / num_edges

        # Add edges for each step
        for k in range(num_edges):
            G_augmented.add_edge(segment[k], segment[k + 1], weight=weight_per_edge)

        i = j
    return G_augmented


# endsection


# section NETWORKX
def nx_to_if7(
    G, first_stop=None, bidirectional=False, bus_name="17 bus", max_travel_time=None
):
    # TODO directionality
    max_round = lambda x: max(1, round(x))
    if max_travel_time:
        max_round = lambda x: min(max_travel_time, max(1, round(x)))
    stops = [str(n) for n in G.nodes]
    if first_stop is None:
        stop1 = stops.pop(0)
    end_str = f"{stop1} is a room. "
    last_stop = stop1
    for s in stops:
        end_str += f"{s} is north of {last_stop}. "
        last_stop = s
    end_str += f"""\n\nThe {bus_name}  is a relatively-scheduled train-car in {stop1}. The waiting duration of the {bus_name}  is 1 minute. The t-schedule of the {bus_name}  is the Table of {bus_name} Schedule.\n"""
    end_str += f"""
Table of {bus_name} Schedule
transit time	destination
"""
    for n in G.edges(data=True):
        end_str += f"{max_round(n[2]['weight'])} minute\t{n[0]}\n"
    # have to get the last one
    n = [n for n in G.edges(data=True)][-1]
    end_str += f"{max_round(n[2]['weight'])} minute\t{n[1]}\n"
    if bidirectional:
        for n in reversed([n for n in G.edges(data=True)]):
            end_str += f"{max_round(n[2]['weight'])} minute\t{n[0]}\n"
    # have to get the last one

    return end_str


def big_nx_to_if7(
    G,
    BIG_G: nx.Graph,
    first_stop=None,
    bidirectional=False,
    bus_name="17 bus",
    max_travel_time=None,
):
    g_pos = nx.spring_layout(BIG_G)
    # TODO directionality
    max_round = lambda x: max(1, round(x))
    if max_travel_time:
        max_round = lambda x: min(max_travel_time, max(1, round(x)))
    stops = [str(n) for n in G.nodes]
    if first_stop is None:
        stop1 = stops.pop(0)
    end_str = f"{stop1} is a room. "
    last_stop = stop1
    for s in range(len(stops)):
        try:
            next_s = stops[s + 1]
        except:
            next_s = stops[s]
        s = stops[s]
        dir = get_direction(*g_pos[s], *g_pos[next_s])
        end_str += f"{s} is {dir} of {last_stop}. "
        last_stop = s
    end_str += f"""\n\nThe {bus_name}  is a relatively-scheduled train-car in {stop1}. The waiting duration of the {bus_name}  is 1 minute. The t-schedule of the {bus_name}  is the Table of {bus_name} Schedule.\n"""
    end_str += f"""
Table of {bus_name} Schedule
transit time	destination
"""
    for n in G.edges(data=True):
        end_str += f"{max_round(n[2]['weight'])} minute\t{n[0]}\n"
    # have to get the last one
    n = [n for n in G.edges(data=True)][-1]
    end_str += f"{max_round(n[2]['weight'])} minute\t{n[1]}\n"
    if bidirectional:
        for n in reversed([n for n in G.edges(data=True)]):
            end_str += f"{max_round(n[2]['weight'])} minute\t{n[0]}\n"
    # have to get the last one

    return end_str


# endsection
def test17():
    # currently some functions are hardcoded
    time_dict = df_to_travel_times(get_df("https://trimet.org/schedules/w/t1009_1.htm"))
    G = time_dict_to_digraph(time_dict)
    # stop_seq = get_stops_from_map(17)
    # G = insert_all_stops(G, stop_seq)
    print(nx_to_if7(G, bidirectional=True))


def making_for_game():
    url_list = {
        "17 bus": "https://trimet.org/schedules/w/t1017_0.htm",
        "9 Bus": "https://trimet.org/schedules/w/t1009_1.htm",
    }
    for name, url in url_list.items():
        time_dict = df_to_travel_times(get_df(url))
        G = time_dict_to_digraph(time_dict)
        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos)
        nx.draw_networkx_labels(G, pos)
        with open(f"{name}.txt", "w+") as f:
            f.writelines(
                nx_to_if7(G, bidirectional=True, bus_name=name, max_travel_time=3)
            )

    plt.show()


def list_to_dg(stop_names, fg):
    for i in range(len(stop_names) - 1):
        start = stop_names[i]
        end = stop_names[i + 1]
        fg.add_edge(start, end, weight=1)
    return fg


def generate_grid_if7():
    stop_names1 = [
        "first street",
        "second ave",
        "third",
        "fourth street",
        "fifth street",
        "sixth street",
    ]

    stop_names2 = ["Park street", "B street", "C street", "D street"]
    zipz = []

    zipz = []
    for street2 in stop_names2:
        combined = [f"{street2} & {street1}" for street1 in stop_names1]
        zipz.append(combined)
    final_rows = []
    for row in zipz:
        new_row = [row[0]]
        for i in range(len(row) - 1):
            room_a = row[i]
            room_b = row[i + 1]
            new_row.append(f"bt{room_a}andbt{room_b}")
            new_row.append(room_b)
        final_rows.append(new_row)
    # now from zipz I want to do the same but columnwise and then combine them

    for i in range(len(zipz) - 1):
        middle_row = []
        for x in range(len(zipz[i])):
            room_a = zipz[i][x]
            room_b = zipz[i + 1][x]
            middle_row.append(f"bt{room_a}andbt{room_b}")
            middle_row.append(None)
        final_rows.insert(i * 2 + 1, middle_row)
    end_str = ""
    for i in final_rows:
        for room in i:
            if room is not None:
                end_str += f"{room} is a room.\n"
    end_str += "\n\n"
    # now for connections
    # just write north east conncections
    for i, row in enumerate(final_rows):
        for x, room in enumerate(row):
            if room is not None:
                west_conn = None
                north_conn = None
                if x > 0:
                    west_conn = row[x - 1]
                if west_conn is not None:
                    end_str += f"{room} is west of {west_conn}.\n"

                if i > 0:
                    north_conn = final_rows[i - 1][x]
                if north_conn is not None:
                    end_str += f"{room} is north of {north_conn}.\n"
    print(end_str)


def multi_network_generator():
    stop_names = [
        "first street",
        "second ave",
        "third",
        "fourth street",
        "fifth street",
        "sixth street",
    ]
    fg = nx.DiGraph()
    fg = list_to_dg(stop_names, fg)
    stop_names = ["Park street", "B street", "second ave", "C street", "D street"]
    fg2 = nx.DiGraph()
    fg2 = list_to_dg(stop_names, fg2)
    stop_names = ["francais ave", "Park street", "marquam ave", "crown park"]
    fg3 = nx.DiGraph()
    fg3 = list_to_dg(stop_names, fg3)
    stop_names = [
        "sixth street",
        "angel & 112th",
        "angel & 114th",
        "angel & 128th",
        "angel plaza",
        "crown park",
    ]
    fg4 = list_to_dg(stop_names, nx.DiGraph())
    cd = {"18 Bus": fg, "Green Line ": fg2, "12 Bus": fg3, "Metro": fg4}
    big_g = nx.compose_all(cd.values())
    global_pos = {}
    with open(f"big.txt", "w+") as f:
        for name, G in cd.items():
            f.writelines(
                big_nx_to_if7(G, big_g, bus_name=name, max_travel_time=3) + "\n"
            )

    pos = nx.spring_layout(big_g)
    nx.draw(big_g, pos)
    nx.draw_networkx_labels(big_g, pos)
    plt.show()


if __name__ == "__main__":
    # multi_network_generator()
    generate_grid_if7()
