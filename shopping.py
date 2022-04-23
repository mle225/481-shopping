import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
# list of months in correct order to get index
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def get_month(month):
    return MONTHS.index(month)

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidences = []
    labels = []

    with open(filename, "r") as raw_data: 
        # use csv dictReader to quickly access data by col name in each row
        data = csv.DictReader(raw_data)
        for row in data: 
            curr_row = []
            # format and append appropriately each field of entry
            curr_row.append(int(row["Administrative"]))
            curr_row.append(float(row["Administrative_Duration"]))
            curr_row.append(int(row["Informational"]))
            curr_row.append(float(row["Informational_Duration"]))
            curr_row.append(int(row["ProductRelated"]))
            curr_row.append(float(row["ProductRelated_Duration"]))
            curr_row.append(float(row["BounceRates"]))
            curr_row.append(float(row["ExitRates"]))
            curr_row.append(float(row["PageValues"]))
            curr_row.append(float(row["SpecialDay"]))
            # convert month by index and add
            month = get_month(row["Month"])
            curr_row.append(int(month))
            curr_row.append(int(row["OperatingSystems"]))
            curr_row.append(int(row["Browser"]))
            curr_row.append(int(row["Region"]))
            curr_row.append(int(row["TrafficType"]))
            # convert visitor type true false to int
            vst_type = int(row["VisitorType"] == 'Returning_Visitor')
            curr_row.append(vst_type)
            # convert weekend true false to int
            wknd = int(row["Weekend"] == 'TRUE')
            curr_row.append(wknd)
            # add current formatted row to evidences
            evidences.append(curr_row)

            # add label
            lbl = int(row["Revenue"] == 'TRUE')
            labels.append(lbl)
            
    # print(labels)

    return evidences, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # set variables for total positives/negatives and true positives/ negatives
    total_neg, total_pos, true_neg, true_pos = 0,0,0,0

    for i in range(len(labels)):

        if labels[i] == 0:
            total_neg += 1
            #prediction is true
            if labels[i] == predictions[i]:
                true_neg += 1
            
        # if label is 1
        else: 
            total_pos += 1
            if labels[i] == predictions[i]:
                true_pos += 1

    # sensitivity = true pos / total pos
    sensitivity = true_pos / total_pos
    # specificity = true neg / total neg
    specificity = true_neg / total_neg

    return sensitivity, specificity
        
if __name__ == "__main__":
    main()


