with open("gbif_nd_codes.txt", "r") as codes, open("all_countries.csv", "r") as countries:

    a = countries.read()

    for line in codes.readlines():
        regions = line.strip("\n").split(", ")
        for region in regions:
            if region in a:
                # print("match {}".format(region))
                None
            else:
                print("no match {}".format(region))
