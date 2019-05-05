from flask import Flask, jsonify
from keras.models import load_model
import tensorflow as tf
import numpy as np
import requests
import datetime as dt
import json

lat_long = np.load("lat_long.npy")
model = load_model("time_model.h5")

graph = tf.get_default_graph()

APIKEY = "42271b066ab626aec8a7dbafcb78a50a"

app = Flask(__name__)
app.config["DEBUG"] = True

# The two primary features are Weather and Time.


@app.route(
    "/<string:airline_code>/<int:year>/<int:month>/<int:date>/<int:day_of_week>",
    methods=["GET"],
)
def home(airline_code, year, month, date, day_of_week):
    # TIME
    global graph
    with graph.as_default():
        data_point = np.zeros(19)
        data_point[month - 1] = 1
        data_point[day_of_week - 1 + 12] = 1

        holder = np.zeros((1, 19))
        for i in range(19):
            holder[0][i] = data_point[i]

        time_output = model.predict(holder)

    # WEATHER
    latitude = 0
    longitude = 0
    satisfied = False
    for (code, lat, long) in lat_long:
        if code == airline_code:
            latitude = lat
            longitude = long
            satisfied = True
    request = (
        "http://api.openweathermap.org/data/2.5/forecast?lat="
        + latitude
        + "&lon="
        + longitude
        + "&appid="
        + APIKEY
    )
    response = requests.get(request)
    jsdata = response.json()

    # Calculate daysfromtoday
    precipitation = 0
    weather_output = []
    dt_year = int(dt.datetime.now().year)
    dt_month = int(dt.datetime.now().month)
    dt_date = int(dt.datetime.now().day)
    if dt_year == year and dt_month == month:
        if (date - dt_date) < 5 and (date - dt_date) >= 0:
            daysfromtoday = date - dt_date

            # Collecting relevant data
            if "rain" in jsdata["list"][daysfromtoday]:
                if "3h" in jsdata["list"][daysfromtoday]["rain"]:
                    precipitation = jsdata["list"][daysfromtoday]["rain"]["3h"]
                else:
                    precipitation = jsdata["list"][daysfromtoday]["rain"]["1h"]
            if "snow" in jsdata["list"][daysfromtoday]:
                if "3h" in jsdata["list"][daysfromtoday]["snow"]:
                    precipitation = jsdata["list"][daysfromtoday]["snow"]["3h"]
                else:
                    precipitation = jsdata["list"][daysfromtoday]["snow"]["1h"]

            weather_output = [
                jsdata["list"][daysfromtoday]["clouds"]["all"],  # clouds
                jsdata["list"][daysfromtoday]["main"]["humidity"],  # humidity
                jsdata["list"][daysfromtoday]["wind"]["speed"],  # windspeed
                precipitation,  # precipitation
            ]
        else:
            weather_output = [-1, -1, -1, -1]
    else:
        weather_output = [-1, -1, -1, -1]

    return jsonify([time_output.tolist(), weather_output])


app.run(port=8080)
