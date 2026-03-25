import requests
from bs4 import BeautifulSoup
import time

urls = [
    # "https://iitj.ac.in/computer-science-engineering/",
    # "https://iitj.ac.in/electrical-engineering/",
    # "https://iitj.ac.in/mechanical-engineering/",
    # "https://iitj.ac.in/civil-and-infrastructure-engineering/",
    #  "https://iitj.ac.in/bioscience-bioengineering",
    #  "https://iitj.ac.in/chemical-engineering/",
    #  "https://iitj.ac.in/materials-engineering/en/materials-engineering",
    #  "https://iitj.ac.in/mathematics/",
    #  "https://iitj.ac.in/physics/",
    #  "https://iitj.ac.in/chemistry/en/chemistry"
    "https://iitj.ac.in/PageImages/Gallery/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf"
]

for url in urls:
    print(f"Scraping: {url}")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles (cleaner text 🔥)
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()

    # Clean spacing
    lines = [line.strip() for line in text.splitlines()]
    text = " ".join(line for line in lines if line)

    # Append to same file
    with open("regulations.txt", "a", encoding="utf-8") as f:
        f.write("\n" + text)

    time.sleep(2)  # avoid hitting server too fast

print("All data collected successfully!")