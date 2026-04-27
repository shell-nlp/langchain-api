import styles from '../ChatInterface.module.css'

interface PaginationProps {
  page: number
  pageTotal: number
  total: number
  onPrev: () => void
  onNext: () => void
}

export function Pagination({
  page,
  pageTotal,
  total,
  onPrev,
  onNext,
}: PaginationProps) {
  return (
    <div className={styles.managementPagination}>
      <button
        className={styles.managementMinorButton}
        onClick={onPrev}
        disabled={page <= 1}
      >
        上一页
      </button>
      <span className={styles.managementPaginationInfo}>
        第 {page} / {pageTotal} 页，共 {total} 条
      </span>
      <button
        className={styles.managementMinorButton}
        onClick={onNext}
        disabled={page >= pageTotal}
      >
        下一页
      </button>
    </div>
  )
}
