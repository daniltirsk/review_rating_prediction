import requests
import os

BASE = "urlHere"

reviews = [
    "i think this movie is really bad",
    "dying in a fire accident would feel better than watching this movie",
    "My girlfriend really liked the main actor and the plot wasn't so bad either",
    "It's a modern classic, a true masterpiece, i would never get tired of watching it",
    "The crappiest movie i have ever seen"
]

response = requests.post(BASE+"/get_ratings",json={'reviews':reviews,'method':'both'})

print(response)
print(response.json())