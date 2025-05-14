# section TRIMET BUS THINGS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

from datetime import datetime
import networkx as nx
import requests
import pandas as pd


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
        with open(f"{name}.txt", "w+") as f:
            f.writelines(
                nx_to_if7(G, bidirectional=True, bus_name=name, max_travel_time=3)
            )


if __name__ == "__main__":
    making_for_game()
