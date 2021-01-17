# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:latest

# recommended by python
ENV PYTHONUNBUFFERED 1

# create new user in container (default user would be root)
# and you want to be able to access the code on your host machine as a non-root user
# (automaticaly creates home dir for new user)
RUN useradd -ms /bin/bash ben

# Copy the requirements.txt file into the container at your new users home dir
COPY requirements.txt /home/ben

# Change ownsership of requirements.txt to newuser
# because you want everything installed for new user not for root
RUN chown ben:ben /home/ben/requirements.txt

# Switch to new user
USER ben

# Set the working directory to the new users home dir
WORKDIR /home/ben

# Create mount dir
# IMPORTANT: mount dir has to be empty!!!
# otherwise files will disapear and you cant use successfully installed python modules anymore
RUN mkdir test_dir

# Install any needed packages specified in requirements.txt for the new user
RUN pip install --user --no-cache-dir -r requirements.txt

# add /home/ben/.local/bin to path (tackles the warning that appears during build process)
ENV PATH="/home/ben/.local/bin:${PATH}"

# Make port 8080 available to the world outside this container
# You cant use port 80 because you arent root
#EXPOSE 8080

# Run djangos development server
# Notice the path of manage.py and the correct port
#CMD ["python", "basic_chat/basic_chat/manage.py", "runserver", "0.0.0.0:8080"]

