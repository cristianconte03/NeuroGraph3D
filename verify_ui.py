from playwright.sync_api import sync_playwright, expect
import os
import re

def verify_neurograph_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        cwd = os.getcwd()

        try:
            # 1. Navigate to Home
            print("Navigating to Home...")
            page.goto("http://127.0.0.1:8050/")

            # Wait for content to load
            page.wait_for_selector("#upload-mapping", timeout=10000)

            # Screenshot Home (Upload)
            print("Taking screenshot of Home...")
            page.screenshot(path=os.path.join(cwd, "verification_home.png"), full_page=True)

            # 2. Check Visual Elements on Home
            expect(page.get_by_text("NeuroGraph 3D").first).to_be_visible()

            # Check ID
            upload_element = page.locator("#upload-mapping")
            expect(upload_element).to_be_visible()

            # Check Class
            # Note: Dash might append classes or use dash-upload-component class
            class_attr = upload_element.get_attribute("class")
            print(f"Upload element classes: {class_attr}")

            if "upload-box" not in (class_attr or ""):
                print("WARNING: .upload-box class not found on #upload-mapping")
                # Print outer HTML to debug
                print(upload_element.evaluate("el => el.outerHTML"))

            # 3. Navigate to Compare
            print("Navigating to Compare...")
            page.get_by_role("tab", name="Confronto Reti").click()
            page.wait_for_timeout(2000)

            # Screenshot Compare
            print("Taking screenshot of Compare...")
            page.screenshot(path=os.path.join(cwd, "verification_compare.png"), full_page=True)

            # 4. Navigate to Simulation
            print("Navigating to Simulation...")
            page.get_by_role("tab", name="Simulazione AI").click()
            page.wait_for_timeout(2000)

            # Screenshot Simulation
            print("Taking screenshot of Simulation...")
            page.screenshot(path=os.path.join(cwd, "verification_simulation.png"), full_page=True)

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path=os.path.join(cwd, "verification_error.png"))
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_neurograph_ui()
