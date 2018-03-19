#!/bin/bash
for i in {160..196}
do
  nmap 140.233.$i.0/24 -sL | grep raspberrypi
done
