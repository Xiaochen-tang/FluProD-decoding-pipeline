import cv2
import numpy as np
import pandas as pd

# Barcode definitions
codes = {
    "code1": 0.00,
    "code2": 0.17,
    "code3": 0.33,
    "code4": 0.50,
    "code5": 0.67,
    "code6": 0.83,
    "code7": 1.00
}


def decode(ratio):
    """
    Assign barcode based on nearest predefined ratio.
    """
    distances = {}

    for k, v in codes.items():
        distances[k] = abs(ratio - v)

    return min(distances, key=distances.get)


def run_decoding(fam_path, cy3_path):

    # Step 1 Read images
    fam = cv2.imread(
        fam_path,
        cv2.IMREAD_GRAYSCALE
    )

    cy3 = cv2.imread(
        cy3_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Step 2 Background removal
    _, fam_thresh = cv2.threshold(
        fam,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, cy3_thresh = cv2.threshold(
        cy3,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3 Size filtering
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            fam_thresh
        )

    mask = np.zeros_like(fam)

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if 50 < area < 5000:
            mask[labels == i] = 255

    # Step 4 ROI extraction
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(
            cnt
        )

        roi_fam = fam[
            y:y+h,
            x:x+w
        ]

        roi_cy3 = cy3[
            y:y+h,
            x:x+w
        ]

        # Step 5 Intensity quantification
        fam_intensity = np.mean(
            roi_fam
        )

        cy3_intensity = np.mean(
            roi_cy3
        )

        # Step 6 Ratio calculation
        ratio = cy3_intensity / (
            cy3_intensity + fam_intensity
        )

        decoded = decode(ratio)

        results.append([
            fam_intensity,
            cy3_intensity,
            ratio,
            decoded
        ])

    df = pd.DataFrame(
        results,
        columns=[
            "FAM",
            "Cy3",
            "Ratio",
            "Decoded"
        ]
    )

    # Step 7 Total intensity grouping
    df["TotalIntensity"] = \
        df["FAM"] + df["Cy3"]

    bins = [0, 100, 300, 600, 10000]

    labels = [
        "Low",
        "Medium",
        "High",
        "VeryHigh"
    ]

    df["IntensityGroup"] = pd.cut(
        df["TotalIntensity"],
        bins=bins,
        labels=labels
    )

    return df
