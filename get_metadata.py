import uuid
import datetime
import geocoder

# Generate a UUID
uuid_value = uuid.uuid4()

# Print the UUID
print(uuid_value)

# Get the current time with UTC+8
current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)

# Print the current time
print(current_time)

# Get latitude and longitude
g = geocoder.ip('me')
latitude = g.latlng[0]
longitude = g.latlng[1]

# Print latitude and longitude
print("Latitude:", latitude)
print("Longitude:", longitude)