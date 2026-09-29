export function getYoutubeId(url) {
    if (!url) return null;
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
}


export function extractMedia(data) {

    if (!data) return {}

    const urls = data.urls || []
    const images = data.images || []

    let youtubeId = null

    for (const u of urls) {
        const id = getYoutubeId(u)
        if (id) {
            youtubeId = id
            break
        }
    }

    return {
        urls,
        images,
        youtubeId,
    }
}