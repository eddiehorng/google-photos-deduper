import logging, time
from app.lib.media_items_image_store import MediaItemsImageStore
from app import config


class DuplicateFilenameDetector:
    def __init__(
        self,
        media_items: list[dict],
        logger=logging.getLogger(),
    ):
        self.media_items = media_items
        self.logger = logger

    # similarity_map: dict[image_id1][image_id2] = score
    # groups: [[idx_1, idx_2, ..], [...]] => list of same fn list
    def calculate_similarity_map(self):
        similarity_map = {}
        groups = []
        start = time.perf_counter()
        # Convert these into a dict of dict[image_id1][image_id2] = score
        for idx, m in enumerate(self.media_items):
            m = self.media_items[idx]
            # self.logger.info(f"calculate_similarity_map: {m}")
            s, g = self.find_same_filename(m)
            similarity_map[m['id']] = s
            if s:
                g.append(idx)
                groups.append(g)

            if idx%100==0:
                self.logger.info(f"processed {idx}: {idx*100/len(self.media_items):.1f}%")

        self.logger.info(
            f"Calculated similarity map in {(time.perf_counter() - start):.2f} seconds"
        )            
        groups = sorted(groups, key=lambda x: len(x), reverse=True)        
        return similarity_map, groups 

    # compare item against all other items
    # return {id: score}
    def find_same_filename(self, item): 
        score_dict = {}
        group = []
        for idx, m in enumerate(self.media_items):
            m = self.media_items[idx]
            same = item['filename']==m['filename'] and item['mediaMetadata']['creationTime']==m['mediaMetadata']['creationTime']
            if item['id'] != m['id'] and same:                 
                score_dict[m['id']] = 1
                group.append(idx)
        return score_dict, group


    def calculate_groups(self):
        embeddings = self._calculate_embeddings()

        start = time.perf_counter()
        groups = self._community_detection(
            embeddings,
            min_community_size=2,
            threshold=self.threshold,
        )
        self.logger.info(
            f"Calculated groups in {(time.perf_counter() - start):.2f} seconds"
        )

        return groups

   