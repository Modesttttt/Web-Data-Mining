class CleanDataPipeline:
    def process_item(self, item):
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = value.strip()
        return item
