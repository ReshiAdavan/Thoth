use indexmap::IndexMap;

///////////////////////////////////////////////////////////////
/////////////////// SHARED HELPER FUNCTIONS ///////////////////
///////////////////////////////////////////////////////////////

pub fn count_common_encoded_tuples(encoded_ints: Vec<usize>) -> IndexMap<(usize, usize), usize> {
    let mut freq_map: IndexMap<(usize, usize), usize> = IndexMap::new();
    for window in encoded_ints.windows(2) {
        if let [a, b] = *window {
            *freq_map.entry((a.into(), b.into())).or_insert(0) += 1;
        }
    }
    freq_map
}

pub fn merge(pair: (usize, usize), ids: Vec<usize>, idx: usize) -> Vec<usize> {
    let mut merged_ids = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && (ids[i], ids[i + 1]) == pair {
            merged_ids.push(idx);
            i += 2;
        } else {
            merged_ids.push(ids[i]);
            i += 1;
        }
    }
    merged_ids
}

pub fn replace_control_characters(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() {
                format!("\\u{:04x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}

pub fn render_token(t: &[u8]) -> String {
    let decoded = String::from_utf8_lossy(t);
    replace_control_characters(&decoded)
}