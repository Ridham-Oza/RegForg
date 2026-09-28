from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import ( CHUNK_SIZE, CHUNK_OVERLAP)


def split_documents(documents):
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    
    
    seprators=["\n\n", "\n", " ", ""],
    
    )


    chunks = splitter.split_documents(documents)

    #add chunks ID

    for index, chunk in enumerate(chunks):
        
        chunk.metadata["chunk_id"] = index

        if "page" in chunk.metadata:
            chunk.metadata["display_pages"] = (int(chunk.metadata["pages"])  +  1)

        else:
            chunk.metadata["display_pages"] = "N/A"

    return chunks