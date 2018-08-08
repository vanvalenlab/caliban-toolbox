#!/usr/bin/env python
import os
from jinja2 import Template


def main():

    home = os.environ["HOME"]

    db_host = raw_input("Database host (e.g., ds123456.mlab.com): ")
    db_name = raw_input("Database name (e.g., celltk): ")
    db_username = raw_input("Database username (e.g., fireworks): ")
    db_password = raw_input("Database password (stored in plaintext, unfortunately): ")
    db_port = raw_input("Database port (e.g. 99999 from ds123456.mlab.com:99999/celltk): ")

    celltk_path = raw_input("celltk path (e.g., %s): " % os.path.join(home, "CellTK"))
    logdir_launchpad = raw_input("Launchpad logging directory (default. %s): " % os.path.join(home, "fw", "logs", "launchpad"))
    logdir_qadapter = raw_input("Queue adapter logging directory (default. %s): " % os.path.join(home, "fw", "logs", "qadapter"))

    if logdir_launchpad == "":
        logdir_launchpad = os.path.join(home, "fw", "logs", "launchpad")
    if logdir_qadapter == "":
        logdir_qadapter = os.path.join(home, "fw", "logs", "qadapter")

    template_my_launchpad = os.path.join(celltk_path, "fireworks", "template", "my_launchpad.yaml.template")
    my_launchpad = os.path.join(celltk_path, "fireworks", "my_launchpad.yaml")

    os.makedirs(logdir_launchpad)
    os.makedirs(logdir_qadapter)

    h = open(template_my_launchpad, "r")
    t = Template(h.read())
    h.close()

    my_launchpad_text = t.render({
        "LOGDIR_LAUNCHPAD": logdir_launchpad,
        "DB_HOST": db_host,
        "DB_NAME": db_name,
        "DB_USERNAME": db_username,
        "DB_PASSWORD": db_password,
        "DB_PORT": db_port,
        })

    h = open(my_launchpad, "w")
    h.write(my_launchpad_text)
    h.close()

    template_my_qadapter = os.path.join(celltk_path, "fireworks", "template", "my_qadapter.yaml.template")
    my_qadapter = os.path.join(celltk_path, "fireworks", "my_qadapter.yaml")

    h = open(template_my_qadapter, "r")
    t = Template(h.read())
    h.close()

    my_qadapter_text = t.render({
        "LOGDIR_QADAPTER": logdir_qadapter,
        "CELLTK_PATH": os.path.join(celltk_path, "fireworks"),
        })

    h = open(my_qadapter, "w")
    h.write(my_qadapter_text)
    h.close()

    print "Created %s with the information provided." % my_launchpad
    print "Created %s with the information provided." % my_qadapter

if __name__ == "__main__":
    main()
