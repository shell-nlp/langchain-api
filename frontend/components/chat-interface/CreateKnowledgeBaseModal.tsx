import styles from '../ChatInterface.module.css'

interface CreateKnowledgeBaseModalProps {
  open: boolean
  knowledgeBaseName: string
  knowledgeBaseDescription: string
  savingKnowledgeBase: boolean
  onClose: () => void
  onNameChange: (value: string) => void
  onDescriptionChange: (value: string) => void
  onCreate: () => void | Promise<void>
}

export function CreateKnowledgeBaseModal({
  open,
  knowledgeBaseName,
  knowledgeBaseDescription,
  savingKnowledgeBase,
  onClose,
  onNameChange,
  onDescriptionChange,
  onCreate,
}: CreateKnowledgeBaseModalProps) {
  if (!open) return null

  return (
    <div className={styles.managementModalOverlay} onClick={onClose}>
      <div className={styles.managementModal} onClick={(event) => event.stopPropagation()}>
        <div className={styles.managementHeader}>
          <h3>新建知识库</h3>
          <button
            type="button"
            className={styles.managementModalClose}
            onClick={onClose}
          >
            关闭
          </button>
        </div>
        <div className={styles.managementForm}>
          <input
            className={styles.managementInput}
            value={knowledgeBaseName}
            onChange={(event) => onNameChange(event.target.value)}
            placeholder="知识库名称"
            autoFocus
          />
          <input
            className={styles.managementInput}
            value={knowledgeBaseDescription}
            onChange={(event) => onDescriptionChange(event.target.value)}
            placeholder="知识库描述"
          />
          <div className={styles.managementToolbar}>
            <button
              className={styles.managementButton}
              disabled={savingKnowledgeBase}
              onClick={() => void onCreate()}
            >
              创建知识库
            </button>
            <button
              type="button"
              className={styles.managementMinorButton}
              onClick={onClose}
            >
              取消
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
