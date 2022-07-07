import numpy as np
from transformers import QDQBertForNextSentencePrediction
arr = np.random.randint(low=0,high=100,size=(2))

def partition(arr,left, right):
    pivot = arr[left]
    i = left + 1
    index = left + 1
    while i< right:
        if arr[i]< pivot:
            arr[i], arr[index] = arr[index], arr[i]
            index += 1
        i += 1
    arr[left],arr[index-1] = arr[index-1] , arr[left]
    return index - 1

def qsort(arr,left,right):
    if right-left <= 1:
        return 
    partitions = partition(arr, left,right)
    qsort(arr,left,partitions)
    qsort(arr,partitions+1,right)

qsort(arr,0,arr.shape[0])
print(arr)