from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import yaml

# NOT WORKING RIGHT NOW

# Set up the Selenium WebDriver
driver = webdriver.Chrome()

# Navigate to the page with dynamic content
driver.get("https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard")

# Wait for the iframe to load and switch to it
try:
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "iframe"))
    )
    iframe = driver.find_element(By.TAG_NAME, "iframe")
    driver.switch_to.frame(iframe)

    print("Switched to iframe")

    # Once the iframe is switched, wait for all tables to be present
    WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table"))
    )

    # Select the second table
    tables = driver.find_elements(By.CSS_SELECTOR, "table")
    # Make sure there are at least two tables found
    if len(tables) > 1:
        second_table = tables[1]  # This is the second table
    else:
        print("Second table not found")
        driver.quit()
        exit(1)

    print("Table is present")

    # Proceed with finding tbody within the second table
    tbody = second_table.find_element(By.TAG_NAME, "tbody")

    # Now find all rows within the second table's tbody
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    # Extract the data
    data = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        row_data = []
        for col in cols:
            cell_wrap = col.find_element(By.CLASS_NAME, "cell-wrap")
            spans = cell_wrap.find_elements(By.TAG_NAME, "span")
            for span in spans:
                text = span.text.strip()
                if text:  # Checking if text is not empty
                    row_data.append(text)
                    break  # Assuming we want the first non-empty text
        data.append(row_data)

    # Convert the list of data to a YAML string
    yaml_data = yaml.dump(data)

finally:
    driver.quit()  # Close the browser

# Output the data to the console (or write to a file)
print(yaml_data)

# Write the YAML data to a file
with open("output.yaml", "w") as file:
    file.write(yaml_data)
