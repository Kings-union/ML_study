import pandas

food_info = pandas.read_csv("food_info.csv")
# print(type(food_info))
# print(food_info.dtypes)
# print(food_info)

# display the first n line of the sample
# print(food_info.head(0))

# display the last n line of the sample
# print(food_info.tail(4))

# display the sample columns
# print(food_info.columns)

# check the sample structure
# print(food_info.shape)

# display the row x data
# print(food_info.loc[0])
# print(food_info.loc[3:6])

# display the column x data
# ndb_col = food_info["NDB_No"]
# print(ndb_col)
# columns = ["Zinc_(mg)", "Copper_(mg)"]
# zinc_copper = food_info[columns]
# print(zinc_copper)

# display the columns end with charactor g
# col_names = food_info.columns.tolist()
# print(col_names)
# gram_columns = []
#
# for c in col_names:
#     if c.endswith("g"):
#         gram_columns.append(c)
#
# gram_df = food_info[gram_columns]
# print(gram_df.head(3))

# calculate the columns
# print(food_info["Iron_(mg)"])
# div_1000 = food_info["Iron_(mg)"]/1000
# print(div_1000)

# water_energy = food_info["Water_(g)"] * food_info["Energ_Kcal"]
# iron_grams = food_info["Iron_(mg)"] / 1000
# print(food_info.shape)
# food_info["Iron_(g)"] = iron_grams
# print(food_info.shape)

# max_calories = food_info["Energ_Kcal"].max()
# print(max_calories)
# normalized_calories = food_info["Energ_Kcal"] / max_calories
# normalized_protein = food_info["Protein_(g)"] / food_info["Protein_(g)"].max()
# normalized_fat = food_info["Lipid_Tot_(g)"] / food_info["Lipid_Tot_(g)"].max()
# food_info["Normalized_Protein"] = normalized_protein
# food_info["Normalized_Fat"] = normalized_fat
#
# print(food_info["Normalized_Protein"])
# print(food_info["Normalized_Fat"])

#sort
# food_info.sort_values("Sodium_(mg)", inplace=True)
# print(food_info["Sodium_(mg)"])
#
# food_info.sort_values("Sodium_(mg)", inplace=True, ascending=False)
# print(food_info["Sodium_(mg)"])

