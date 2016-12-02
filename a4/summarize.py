import json

with open('result1') as :    
        users_n = json.load(data_file)
    return users_n

with open('result2') as data_file:    
        tweet_l = json.load(data_file)
    return tweet_l

with open('ci') as data_file:    
        csi = json.load(data_file)
    return csi

with open('classifyi') as data_file:    
        clsi = json.load(data_file)
    return clsi

print("Number of users collected %d", len(users))
print("Number of tweets collected %d:",len(tweet))
print("Number of communities discovered %d:",csi[0])
print("Average number of users per community:",csi[1])
print("Number of instances per class found:",clsi[0][1])
print("One example from each class:",)
