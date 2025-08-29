#!/usr/bin/env python3

import cgi

print("Content-type: text/html\n")

form = cgi.FieldStorage()
username = form.getvalue("username")
password = form.getvalue("password")

# Dummy check
if username == "admin" and password == "password":
    print("<h2>Login Successful!</h2>")
else:
    print("<h2>Login Failed!</h2>")
