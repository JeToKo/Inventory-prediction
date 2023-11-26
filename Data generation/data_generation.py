import csv
import datetime
import random
import matplotlib.pyplot as plt


def estimate_data(date, time):
    # date: 2033-01-01
    # time: 00:00:00
    a = 50
    b = 100

    a_factor = 1
    b_factor = 1

    if date.month == 1:
        a_factor = 1
        b_factor = 1.2

    elif date.month == 2:
        a_factor = 1
        b_factor = 1.0

    elif date.month == 3:
        a_factor = 1
        b_factor = 1.0

    elif date.month == 4:
        a_factor = 0.5
        b_factor = 0.8

    elif date.month == 5:
        a_factor = 0.5
        b_factor = 0.8

    elif date.month == 6:
        a_factor = 0.5
        b_factor = 0.8

    elif date.month == 7:
        a_factor = 0.5
        b_factor = 0.8

    elif date.month == 8:
        a_factor = 0.5
        b_factor = 0.8

    elif date.month == 9:
        a_factor = 0.5
        b_factor = 0.8

    elif date.month == 10:
        a_factor = 1
        b_factor = 1.0

    elif date.month == 11:
        a_factor = 1.3
        b_factor = 1.3

    elif date.month == 12:
        a_factor = 1.7
        b_factor = 1.5

    if time.hour == 12:
        a -= 20
        b -= 20

    data = random.randint(a*a_factor, b*b_factor)
    print(data)
    return data


def main():
    # open the file in the write mode
    with open('../Neural network/inventory_data', 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f, delimiter=',')
        inventory = []
        dates = []

        # Declare start day
        day = datetime.datetime(2023, 1, 1, 0, 0, 0)

        # Add data until goal day is achieved
        while str(day.date()) != '2030-01-01':
            inv = estimate_data(day.date(), day.time())

            inventory.append(inv)
            dates.append(day)

            writer.writerow([day, inv])
            day += datetime.timedelta(hours=6)

    plt.plot(dates, inventory)
    plt.ylabel('Inventory level')
    plt.xlabel('Year')
    plt.savefig('inventory_graph.png')
    plt.show()


main()
