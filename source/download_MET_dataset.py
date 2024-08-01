import requests
import json
import jsonlines
import random
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from retry import retry
from threading import Lock, local

# Thread-local storage for local counters
thread_local = local()

# Define base URL for the Met's API
base_url = 'https://collectionapi.metmuseum.org/public/collection/v1/'

# Define our search terms and their associated departments
# Use None to search a term in any department, i.e. w/o filtering based on departments
search_terms = [
    ('painting', ["Modern Art", "European Paintings"]),
    ('sculpture', ["Modern Art", "Medieval Art", "European Sculpture and Decorative Arts"]),
]
search_terms = [
    ('painting', None),
    ('sculpture', None),
]


# convert a jsonlines fine to a json file
def jsonl_to_json(input_file, output_file):
    # List to hold all JSON objects
    data = []

    # Read the JSONLines file and parse each line
    with open(input_file, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))

    # Write the collected data as a single JSON array
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=2)


# Function to fetch object data with counter
def fetch_object_data(args):
    object_id, shared_counter, lock = args
    # Initialize the thread-local counter if it doesn't exist
    if not hasattr(thread_local, 'counter'):
        thread_local.counter = 0

    object_response = requests.get(f"{base_url}objects/{object_id}")
    data = object_response.json()

    if 'objectID' in data:
        thread_local.counter += 1

    # Update shared counter every 100 objects
    if thread_local.counter % 100 == 0:
        with lock:
            shared_counter.value += 100
            print(f"Processed {shared_counter.value} objects")

    return data


# Function to download images with counter
@retry(tries=3, delay=2, backoff=2)
def download_image(args):
    image_url, local_path, shared_counter, lock = args

    # Initialize the thread-local counter if it doesn't exist
    if not hasattr(thread_local, 'counter'):
        thread_local.counter = 0

    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(response.content)

    thread_local.counter += 1

    # Update shared counter every 30 images
    if thread_local.counter % 30 == 0:
        with lock:
            shared_counter.value += 30
            print(f"Downloaded {shared_counter.value} images")


# Function to process lines and download images
def process_lines(lines, dataset, shared_counter, lock):
    records = []
    os.makedirs(dataset, exist_ok=True)

    for line in lines:
        data = json.loads(line)
        image_url = data['image']
        local_filename = image_url.split('/')[-1]
        local_path = os.path.join(dataset, local_filename)
        records.append((image_url, local_path, shared_counter, lock, data['description']))

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(download_image, record[:4]): record for record in records}
        for future in as_completed(future_to_url):
            record = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(f'{record[0]} generated an exception: {exc}')

    # Write records to file
    with jsonlines.open(f'{dataset}_output.jsonl', mode='w') as writer:
        for _, local_path, _, _, description in records:
            writer.write({'image': local_path, 'description': description})


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download images from the MET dataset')
    parser.add_argument('--output', type=str, default='MET_output.jsonl', help='Output JSON file')
    parser.add_argument('--train', type=str, default='MET_train', help='Output training directory')
    parser.add_argument('--test', type=str, default='MET_test', help='Output test directory')
    parser.add_argument('--num-ids', type=int, default=2000, help='Max. number of IDs of objects that are tested for download')
    args = parser.parse_args()
    print(f'Output JSON file: {args.output}')
    print(f'Output training directory: {args.train}')
    print(f'Output test directory: {args.test}')
    print(f'Number of images to download per search term: {args.num_ids}')

    all_object_ids = []

    for search_term, allowed_departments in search_terms:
        search_url = f"{base_url}search?q={search_term}"
        print(f"Connecting to: {search_url}")
        response = requests.get(search_url)
        data = response.json()
        object_ids_sample = random.sample(data['objectIDs'], min(args.num_ids, len(data['objectIDs'])))
        all_object_ids.extend([(id, search_term, allowed_departments) for id in object_ids_sample])

    print(f"Selected {len(all_object_ids)} object IDs in total")

    # Shared counter and lock for JSON fetching
    class Counter:
        def __init__(self):
            self.value = 0

    json_counter = Counter()
    json_lock = Lock()

    # Fetch object data in parallel
    print("Fetching selected objects")
    with jsonlines.open(args.output, mode='w') as writer:
        with ThreadPoolExecutor(max_workers=5) as executor:
            for object_data in executor.map(fetch_object_data,
                                            [(id, json_counter, json_lock)
                                             for id, _, _ in all_object_ids]):
                if 'objectID' not in object_data:  # Skip objects without an ID, e.g. 404 results
                    continue

                object_id = object_data['objectID']
                search_term = next((term for id, term, _ in all_object_ids if id == object_id), None)
                allowed_departments = next((deps for id, _, deps in all_object_ids if id == object_id), None)

                if search_term is None and allowed_departments is None:
                    print(f"Skipping object {object_id} due to missing search term and departments")
                    continue

                if (allowed_departments is None or object_data.get('department') in allowed_departments) and object_data['primaryImage']:
                    description_components = [
                        object_data['title'],
                        f"a {object_data['objectName']}" if object_data.get('objectName') else None,
                        f"from the {object_data['culture']}" if object_data.get('culture') else None,
                        f"dating to the {object_data['period']}" if object_data.get('period') else None,
                        object_data['dynasty'] if object_data.get('dynasty') else None,
                        object_data['reign'] if object_data.get('reign') else None,
                        f"({object_data['objectDate']})" if object_data.get('objectDate') else None,
                        f"created by {object_data['artistDisplayName']}" if object_data.get(
                            'artistDisplayName') else None,
                        f"in {object_data['country']}" if object_data.get('country') else None,
                        object_data['region'] if object_data.get('region') else None
                    ]
                    description_components = [component for component in description_components if component is not None]
                    description = ', '.join(description_components) + '.'
                    author_name = object_data['artistDisplayName'] if object_data.get('artistDisplayName') else "Unknown"
                    record = {
                        'image': object_data['primaryImage'],
                        'title': object_data['title'],
                        'description': description,
                        '_author': [{'name': author_name}]
                    }
                    writer.write(record)
    print("JSON fetching completed.")

    # Read lines from METoutput.json file
    with open(args.output, 'r') as f:
        lines = f.readlines()
    print(f"Read {len(lines)} lines from {args.output}")
    # Split into train and test sets
    train_lines, test_lines = train_test_split(lines, test_size=0.20)

    # Shared counter and lock for image downloading
    image_counter = Counter()
    image_lock = Lock()

    # Process training and test sets
    print(f"Getting {len(train_lines)} train images")
    process_lines(train_lines, args.train, image_counter, image_lock)
    print(f"Getting {len(test_lines)} test images")
    process_lines(test_lines, args.test, image_counter, image_lock)

    # convert jsonlines to full JSON file
    json_output = os.path.splitext(args.output)[0] + ".json"
    jsonl_to_json(args.output, json_output)
    print(f"Converted {args.output} to {json_output}")