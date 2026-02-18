def build_prompt(record):
    mapping = {
        "sex": {"0": "female patient", "1": "male patient"},
        "base_morphology": {"0": "pedunculated base", "1": "sessile broad base"},
        "wall_thickening": {"0": "no gallbladder wall thickening", "1": "gallbladder wall thickening present"},
        "echogenicity": {"0": "hyperechoic lesion", "1": "hypoechoic lesion"},
        "location": {"0": "lesion at fundus", "1": "lesion at body", "2": "lesion at neck"}
    }
    age = f"{record['age']} year old"
    long_d = f"long diameter {record['long_diameter']} mm"
    short_d = f"short diameter {record['short_diameter']} mm"
    num = f"{record['number_of_polyps']} polyps"
    sex = mapping["sex"][str(record["sex"])]
    base = mapping["base_morphology"][str(record["base_morphology"])]
    wall = mapping["wall_thickening"][str(record["wall_thickening"])]
    echo = mapping["echogenicity"][str(record["echogenicity"])]
    loc = mapping["location"][str(record["location"])]
    detect = f"polyp detected for {record['detection_time_months']} months"
    text = f"{age} {sex} with {num}, {long_d}, {short_d}, {base}, {wall}, {echo}, {loc}, {detect}."
    return text